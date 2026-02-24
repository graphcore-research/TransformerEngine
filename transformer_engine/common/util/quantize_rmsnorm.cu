/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Modified by Google DeepMind for fused RMSNorm + activation + quantize.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * nvte_quantize_rmsnorm_silu / nvte_quantize_rmsnorm
 *
 * Two-stage implementation:
 *   Stage 1: CUDA kernel applies RMSNorm (using pre-computed inv_rms) and
 *            optionally SiLU activation, writing result to a temp buffer
 *   Stage 2: Calls nvte_quantize on the normalized buffer
 *
 * This avoids modifying TE's deep quantize template machinery while
 * still providing the correct API surface.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include "../common.h"

// ═══════════════════════════════════════════════════════════════════════
// CUDA kernel: RMSNorm + optional SiLU, writes bf16 output
// ═══════════════════════════════════════════════════════════════════════

template<bool APPLY_SILU, int BLOCK_SIZE = 256>
__global__ void rmsnorm_act_kernel(
    const nv_bfloat16* __restrict__ input,       // (M, K)
    const float*       __restrict__ inv_rms,     // (M,)
    const nv_bfloat16* __restrict__ norm_weight, // (K,)
    nv_bfloat16*       __restrict__ output,      // (M, K)
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    float ir = inv_rms[row];
    const nv_bfloat16* row_in  = input  + (size_t)row * cols;
    nv_bfloat16*       row_out = output + (size_t)row * cols;

    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float x = __bfloat162float(row_in[i]);
        float w = __bfloat162float(norm_weight[i]);
        float normed = x * ir * w;

        if constexpr (APPLY_SILU) {
            float sigmoid_val = 1.0f / (1.0f + __expf(-normed));
            normed = normed * sigmoid_val;
        }

        row_out[i] = __float2bfloat16(normed);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Host launcher for rmsnorm + act kernel
// ═══════════════════════════════════════════════════════════════════════

static void launch_rmsnorm_act(
    const void* input, const float* inv_rms, const void* norm_weight,
    void* output, int rows, int cols, bool apply_silu, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    if (apply_silu) {
        rmsnorm_act_kernel<true, BLOCK><<<rows, BLOCK, 0, stream>>>(
            static_cast<const nv_bfloat16*>(input),
            inv_rms,
            static_cast<const nv_bfloat16*>(norm_weight),
            static_cast<nv_bfloat16*>(output),
            rows, cols);
    } else {
        rmsnorm_act_kernel<false, BLOCK><<<rows, BLOCK, 0, stream>>>(
            static_cast<const nv_bfloat16*>(input),
            inv_rms,
            static_cast<const nv_bfloat16*>(norm_weight),
            static_cast<nv_bfloat16*>(output),
            rows, cols);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// C API: nvte_quantize_rmsnorm_silu
//   Fused RMSNorm + SiLU + quantize
// ═══════════════════════════════════════════════════════════════════════

void nvte_quantize_rmsnorm_silu(
    const NVTETensor input,
    NVTETensor output,
    const NVTEQuantizationConfig quant_config,
    const float* inv_rms,
    const void* norm_weight,
    cudaStream_t stream
) {
    NVTE_API_CALL(nvte_quantize_rmsnorm_silu);
    using namespace transformer_engine;

    const auto& in_tensor = *convertNVTETensorCheck(input);
    const auto& in_shape = in_tensor.data.shape;
    NVTE_CHECK(in_shape.size() == 2, "Input must be 2D");

    int rows = static_cast<int>(in_shape[0]);
    int cols = static_cast<int>(in_shape[1]);

    // Allocate temp buffer for normalized+activated data (same shape as input)
    void* temp_buf = nullptr;
    size_t buf_bytes = (size_t)rows * cols * sizeof(nv_bfloat16);
    NVTE_CHECK_CUDA(cudaMallocAsync(&temp_buf, buf_bytes, stream));

    // Stage 1: RMSNorm + SiLU → temp_buf
    launch_rmsnorm_act(
        in_tensor.data.dptr, inv_rms, norm_weight,
        temp_buf, rows, cols, /*apply_silu=*/true, stream);

    // Build a TensorWrapper for the temp buffer so we can call nvte_quantize
    std::vector<size_t> temp_shape = {(size_t)rows, (size_t)cols};
    TensorWrapper temp_tensor(temp_buf, temp_shape, DType::kBFloat16);

    // Stage 2: quantize temp → output
    nvte_quantize_v2(temp_tensor.data(), output, quant_config, stream);

    // Free temp buffer (async)
    NVTE_CHECK_CUDA(cudaFreeAsync(temp_buf, stream));
}

// ═══════════════════════════════════════════════════════════════════════
// C API: nvte_quantize_rmsnorm
//   Fused RMSNorm (no activation) + quantize
// ═══════════════════════════════════════════════════════════════════════

void nvte_quantize_rmsnorm(
    const NVTETensor input,
    NVTETensor output,
    const NVTEQuantizationConfig quant_config,
    const float* inv_rms,
    const void* norm_weight,
    cudaStream_t stream
) {
    NVTE_API_CALL(nvte_quantize_rmsnorm);
    using namespace transformer_engine;

    const auto& in_tensor = *convertNVTETensorCheck(input);
    const auto& in_shape = in_tensor.data.shape;
    NVTE_CHECK(in_shape.size() == 2, "Input must be 2D");

    int rows = static_cast<int>(in_shape[0]);
    int cols = static_cast<int>(in_shape[1]);

    // Allocate temp buffer
    void* temp_buf = nullptr;
    size_t buf_bytes = (size_t)rows * cols * sizeof(nv_bfloat16);
    NVTE_CHECK_CUDA(cudaMallocAsync(&temp_buf, buf_bytes, stream));

    // Stage 1: RMSNorm (no activation) → temp_buf
    launch_rmsnorm_act(
        in_tensor.data.dptr, inv_rms, norm_weight,
        temp_buf, rows, cols, /*apply_silu=*/false, stream);

    // Build temp tensor wrapper
    std::vector<size_t> temp_shape = {(size_t)rows, (size_t)cols};
    TensorWrapper temp_tensor(temp_buf, temp_shape, DType::kBFloat16);

    // Stage 2: quantize temp → output
    nvte_quantize_v2(temp_tensor.data(), output, quant_config, stream);

    // Free temp buffer (async)
    NVTE_CHECK_CUDA(cudaFreeAsync(temp_buf, stream));
}
