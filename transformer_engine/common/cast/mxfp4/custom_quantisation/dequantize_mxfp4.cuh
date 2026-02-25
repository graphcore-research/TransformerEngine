/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_mxfp4.cuh
 *  \brief CUDA kernels to dequantize from MXFP4.
 *
 *  MXFP4: E8M0 block scales, block size 32, native FP4 data.
 *  Adapted from dequantize_nvfp4.cuh.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_MXFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "core_mxfp4.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {
namespace dequantize_kernel {
#if FP4_TYPE_SUPPORTED

/// Dequantize MXFP4 data using E8M0 block scales (block size 32).
///
/// Each block of 32 FP4 elements shares one E8M0 scale.
/// Dequantized value = fp4_value * 2^(E8M0_scale - 127)
///
/// No global amax factor is used (unlike NVFP4 which multiplies by amax/(6*448)).
template <typename OType>
__global__ void __launch_bounds__(512)
    dequantize_mxfp4_kernel(const void *const input, OType *output,
                            const e8m0_t *const scales,
                            const size_t N, const size_t M,
                            const size_t scale_stride) {
  constexpr int BLOCK_SIZE = 32;  // MXFP4 block size

  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread processes 16 FP4 elements (= one fp4e2m1x4 vec × 4 iterations)
  const size_t x = thread_idx % M;
  const size_t y = thread_idx / M;

  if (y >= N) {
    return;
  }

  union fp4vec {
    uint64_t vec;
    fp4e2m1x4 small_vec[4];
  };
  using OVec = Vec<OType, 4>;
  const uint64_t *const input_vectorized = reinterpret_cast<const uint64_t *>(input);
  OVec *output_vec = reinterpret_cast<OVec *>(output);

  const size_t my_index = x + y * M;
  const size_t my_output_index = (x + y * M) * 4;
  fp4vec value;
  value.vec = input_vectorized[my_index];

  // Each thread processes 16 elements, which may span across block-32 boundaries.
  // Compute the global column index of the first element:
  const size_t first_col = x * 16;  // Each M-index covers 16 FP4 elements

#pragma unroll
  for (int i = 0; i < 4; i++) {
    float4 current = static_cast<float4>(value.small_vec[i]);
    OVec out;

    // Each of the 4 elements in this sub-group
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const size_t col = first_col + i * 4 + j;
      const size_t scale_col_idx = col / BLOCK_SIZE;
      const size_t scale_idx = y * scale_stride + scale_col_idx;
      e8m0_t scale_byte = scales[scale_idx];
      float final_scale = core::exp2f_e8m0(scale_byte);

      float val;
      switch (j) {
        case 0: val = current.x; break;
        case 1: val = current.y; break;
        case 2: val = current.z; break;
        case 3: val = current.w; break;
      }
      out.data.elt[j] = static_cast<OType>(val * final_scale);
    }
    output_vec[my_output_index + i] = out;
  }
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace dequantize_kernel

/// Host wrapper for MXFP4 dequantization.
inline void dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace dequantize_kernel;
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output");
  NVTE_CHECK(input.data.dtype == DType::kFloat4E2M1, "Input must have FP4 type.");
  NVTE_CHECK(is_high_precision_dtype(output->data.dtype), "Output must be higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes must match.");

  constexpr int FP4_BLOCK_SIZE = 32;  // MXFP4
  const size_t N = input.flat_first_dim();
  const size_t M = input.flat_last_dim();

  NVTE_CHECK(M % FP4_BLOCK_SIZE == 0,
             "Last dimension must be divisible by ", FP4_BLOCK_SIZE);

  // Each uint64_t read covers 16 FP4 elements
  const size_t Mread = M / 16;
  const size_t total = N * Mread;
  const size_t threads = 512;
  const size_t blocks = DIVUP(total, threads);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      output->data.dtype, OType,
      dequantize_mxfp4_kernel<<<blocks, threads, 0, stream>>>(
          input.data.dptr, reinterpret_cast<OType *>(output->data.dptr),
          reinterpret_cast<e8m0_t *>(input.scale_inv.dptr),
          N, Mread, input.scale_inv.shape.back());
  );  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8+ is needed for FP4 support!");
#endif
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_MXFP4_CUH_
