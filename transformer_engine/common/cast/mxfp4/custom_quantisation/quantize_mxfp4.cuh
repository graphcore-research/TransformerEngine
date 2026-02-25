/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp4.cuh
 *  \brief CUDA kernels to cast to MXFP4 (quantize-only, no transpose).
 *
 *  MXFP4: E8M0 block scales, block size 32, native FP4 output.
 *  Adapted from quantize_nvfp4.cuh.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "core_mxfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {
namespace quantize_kernel {

using namespace ptx;
using namespace core;

// MXFP4 block size = 32 (vs NVFP4's 16)
constexpr size_t SCALE_DIM = 32;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t BUFF_DIM_Y = 32;

constexpr size_t PACK_SIZE = 8;
constexpr size_t WAVES = SCALE_DIM / PACK_SIZE;  // 4 (vs NVFP4's 2)

// Number of 4-bit elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256

// Number of threads (rowwise scaling) that span 32 banks
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM;  // 8

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool ENCODE_CENTRIC,
          size_t CHUNK_DIM_Y, size_t CHUNK_DIM_X, size_t THREADS_PER_CHUNK>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_mxfp4_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                          const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
                          mxfp4_scale_t *const scales_rowwise_e8m0,
                          const float *noop,
                          const size_t rows,
                          const size_t cols,
                          const size_t scale_stride_rowwise,
                          const bool swizzle_scales = false) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT =
      (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

  using IType2 = typename ptx::FPx2<IType>;

  if constexpr (!COMPUTE_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }

  // MXFP4 uses block-32, so scaling factors per chunk row
  constexpr size_t SCALING_FACTORS_PER_CHUNK_ROW = CHUNK_DIM_X / SCALE_DIM;
  constexpr size_t THREADS_X_ROWWISE = SCALING_FACTORS_PER_CHUNK_ROW;
  constexpr size_t THREADS_Y_ROWWISE = THREADS_PER_CHUNK / THREADS_X_ROWWISE;

  static_assert(BUFF_DIM_Y >= SCALE_DIM,
                "Number of buffer rows must be >= scale block size");
  static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
  static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE,
                "Buffer rows must be >= rowwise thread count");

  constexpr size_t BUFF_IN_DIM_X = CHUNK_DIM_X;
  constexpr size_t BUFF_OUT_DIM_X = (CHUNK_DIM_X * 4) / 8;  // FP4: 0.5 byte per element
  constexpr size_t BUFF_IN_DIM = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t BUFF_OUT_DIM = BUFF_DIM_Y * BUFF_OUT_DIM_X;

  constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
  constexpr size_t ITERATIONS_ROWWISE = BUFF_DIM_Y / THREADS_Y_ROWWISE;

  const int block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * CHUNK_DIM_X;
  const int scales_block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int scales_block_offset_X = blockIdx.x * CHUNK_DIM_X / SCALE_DIM;

  const int tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const int tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;

  const int thread_offset_Y_rowwise = tid_Y_rowwise;
  const int thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;

  const int row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;

  const int scales_offset_Y_rowwise = scales_block_offset_Y + tid_Y_rowwise;
  const int scales_offset_X_rowwise = scales_block_offset_X + tid_X_rowwise;

  const bool rowwise_scale_is_within_bounds =
      scales_offset_X_rowwise < (int)(cols / SCALE_DIM);

  // Bank conflict resolution
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

  constexpr size_t in_mem = buff_size_aligned_in;
  constexpr size_t out_mem_data = buff_size_aligned_out;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);

  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;
  const bool is_master_thread = (threadIdx.x == 0);

  float thread_amax = 0.0f;

  // Initialize barriers
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_PER_CHUNK>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y,
                    shmem_buff_size, &mbar[0], is_master_thread);

#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const int buff = stage % BUFFS_NUM;
    const int next_stage = stage + 1;
    const int stage_offset_Y = stage * BUFF_DIM_Y;

    const int buff_offset_in = buff * BUFF_IN_DIM;
    const int buff_offset_out = buff * BUFF_OUT_DIM;

    if (next_stage < STAGES) {
      ptx::cp_async_bulk_wait_group_read<1>();

      const int next_buff = next_stage % BUFFS_NUM;
      const int next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const int global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int next_buff_offset = next_buff * BUFF_IN_DIM;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                        global_offset_Y, shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;

    // Rowwise scaling with block size 32
    {
      const int stage_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (int it = 0; it < ITERATIONS_ROWWISE; ++it) {
        const int it_thread_offset_Y = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const int shmem_offset_base_in =
            buff_offset_in + it_thread_offset_Y * BUFF_IN_DIM_X;
        const int shmem_offset_base_out =
            buff_offset_out + it_thread_offset_Y * BUFF_OUT_DIM_X;

        const int it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;
        float in_compute[SCALE_DIM];
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        // 1. Read elements and find block AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const int shmem_offset = shmem_offset_base_in + swizzled_thread_idx;
            in_IType[w].load_from(&in_sh[shmem_offset]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
            }
          }
          block_amax =
              static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const int shmem_offset = shmem_offset_base_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            in.load_from(&in_sh[shmem_offset]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE; ++e) {
              const int j = w * PACK_SIZE + e;
              float elt = static_cast<float>(in.data.elt[e]);
              if constexpr (COMPUTE_ACTIVATIONS) {
                elt = OP(elt, {});
              }
              if constexpr (!std::is_same_v<IType, float>) {
                elt = static_cast<float>(static_cast<IType>(elt));
              }
              if constexpr (COMPUTE_ACTIVATIONS) {
                const bool row_oob = (row_base_rowwise + it_offset_Y >= rows);
                const bool col_oob = (block_offset_X + swizzled_thread_idx >= cols);
                if (!row_oob && !col_oob) {
                  block_amax = fmaxf(block_amax, fabsf(elt));
                }
              } else {
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
              in_compute[j] = elt;
            }
          }
        }

        // 2. Compute E8M0 scaling factor
        mxfp4_scale_t S_b_e8m0;
        float block_scale_inverse;

        if constexpr (ENCODE_CENTRIC) {
          // Encode-centric: compute multiplier, store flipped
          mxfp4_scale_t mult_bits = compute_encoding_scaling_factor(block_amax);
          block_scale_inverse = exp2f_e8m0(mult_bits);
          // Flip for storage (GEMM expects divisor)
          int flipped = 254 - (int)mult_bits;
          S_b_e8m0 = static_cast<mxfp4_scale_t>(max(0, min(255, flipped)));
        } else {
          // Decode-centric (standard MX convention)
          S_b_e8m0 = compute_decoding_scaling_factor(block_amax);
          block_scale_inverse = exp2f_rcp_e8m0(S_b_e8m0);
        }

        // Store scale directly to global memory
        if (rowwise_scale_is_within_bounds) {
          const int scales_Y =
              scales_offset_Y_rowwise + stage_scales_offset_Y + it * THREADS_Y_ROWWISE;
          const int scales_X = scales_offset_X_rowwise;

          if (swizzle_scales) {
            // Write scales in MMA-swizzled layout for TK GEMM.
            // Layout: 128-row × 4-scale tiles (512 bytes each).
            // Scale swizzle for block-32 / E8M0:
            //   tile_m = row / 128, row_in_tile = row % 128
            //   tile_k = (col/32) / 4
            //   k_byte = (col/32) % 4
            //   j = row_in_tile % 32, group = row_in_tile / 32
            //   byte_in_tile = j * 16 + group * 4 + k_byte
            const int tile_m = scales_Y / 128;
            const int row_in_tile = scales_Y % 128;
            const int tile_k = scales_X / 4;
            const int k_byte = scales_X % 4;
            const int j = row_in_tile % 32;
            const int group = row_in_tile / 32;
            const int num_tiles_k = static_cast<int>(scale_stride_rowwise) / 4;
            const int tile_start = (tile_m * num_tiles_k + tile_k) * 512;
            const int byte_in_tile = j * 16 + group * 4 + k_byte;
            reinterpret_cast<uint8_t *>(scales_rowwise_e8m0)[tile_start + byte_in_tile] =
                reinterpret_cast<const uint8_t &>(S_b_e8m0);
          } else {
            const int scale_idx = scales_Y * scale_stride_rowwise + scales_X;
            scales_rowwise_e8m0[scale_idx] = S_b_e8m0;
          }
        }

        // 3. Quantize elements to FP4
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 4; ++e) {
            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              IType2 in01 = in_IType[w].data.elt[2 * e];
              IType2 in23 = in_IType[w].data.elt[2 * e + 1];
              fp4e2m1x4 &out_quad = reinterpret_cast<fp4e2m1x4 &>(out.data.elt[e]);
              ptx::mul_cvt_4x(out_quad, in01, in23, block_scale_inverse);
            } else {
              const int j = w * PACK_SIZE + 4 * e;
              IType2 in01;
              IType2 in23;
              in01.x = in_compute[j];
              in01.y = in_compute[j + 1];
              in23.x = in_compute[j + 2];
              in23.y = in_compute[j + 3];
              fp4e2m1x4 &out_quad = reinterpret_cast<fp4e2m1x4 &>(out.data.elt[e]);
              ptx::mul_cvt_4x(out_quad, in01, in23, block_scale_inverse);
            }
          }
          const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const int swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          const int shmem_offset = shmem_offset_base_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset]);
        }
      }
    }

    __builtin_assume(thread_amax >= 0);
    __builtin_assume(block_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    // TMA store to global
    if (is_master_thread) {
      const int global_offset_Y = block_offset_Y + stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int buff_out_offset = buff * BUFF_OUT_DIM;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise),
          global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_out_offset]));

      ptx::cp_async_bulk_commit_group();
    }
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#endif  // __CUDA_ARCH__ >= 1000
}

}  // namespace quantize_kernel

/// Host-side quantize dispatch.
/// Performs MXFP4 quantization with E8M0 block-32 scales.
inline void quantize(const Tensor &input, const Tensor *noop, Tensor *output,
                     cudaStream_t stream, bool encode_centric = false,
                     bool swizzle_scales = false) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_kernel;
  using namespace ptx;

  constexpr bool COMPUTE_ACTIVATIONS = false;
  using ParamOP = Empty;
  constexpr float (*OP)(float, const ParamOP &) = nullptr;

  NVTE_CHECK(output->has_data(), "MXFP4 output tensor must be allocated.");
  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  if (noop) {
    CheckNoopTensor(*noop, "cast_noop");
  }

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  NVTE_CHECK(rows % 32 == 0, "Number of rows must be a multiple of 32");
  NVTE_CHECK(cols % 32 == 0, "Number of cols must be a multiple of 32");

  constexpr size_t CHUNK_DIM_Y = 128;
  constexpr size_t CHUNK_DIM_X = 128;
  constexpr size_t THREADS_PER_CHUNK = 128;
  constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_PER_CHUNK;

  const size_t scale_stride = output->scale_inv.shape[1];

  mxfp4_scale_t *const scales_ptr =
      reinterpret_cast<mxfp4_scale_t *>(output->scale_inv.dptr);

  const float *noop_ptr = (noop && noop->data.dptr)
                              ? reinterpret_cast<const float *>(noop->data.dptr)
                              : nullptr;

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      alignas(64) CUtensorMap tensor_map_input{};
      alignas(64) CUtensorMap tensor_map_output{};

      create_2D_tensor_map(tensor_map_input, input.data,
                           rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                           cols, 0, sizeof(IType) * 8);

      create_2D_tensor_map(tensor_map_output, output->data,
                           rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                           cols, 0, 4);  // 4 bits per element

      constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
      constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
      constexpr size_t buff_size_aligned_in =
          DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
      constexpr size_t buff_size_aligned_out =
          DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

      constexpr size_t dshmem_size =
          buff_size_aligned_in + buff_size_aligned_out + TMA_SHMEM_ALIGNMENT + 1024;

      #define LAUNCH_MXFP4_QUANTIZE_KERNEL(ENCODE_C) \
        { \
          auto kernel = \
              quantize_mxfp4_kernel<COMPUTE_ACTIVATIONS, ParamOP, OP, IType, \
                                    ENCODE_C, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>; \
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                               dshmem_size); \
          kernel<<<grid, block_size, dshmem_size, stream>>>( \
              tensor_map_input, tensor_map_output, \
              scales_ptr, noop_ptr, rows, cols, scale_stride, swizzle_scales); \
        }

      if (encode_centric) {
        LAUNCH_MXFP4_QUANTIZE_KERNEL(true);
      } else {
        LAUNCH_MXFP4_QUANTIZE_KERNEL(false);
      }
      #undef LAUNCH_MXFP4_QUANTIZE_KERNEL
      NVTE_CHECK_CUDA(cudaGetLastError());
  );  // TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_
