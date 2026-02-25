/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_transpose_mxfp4.cuh
 *  \brief CUDA kernels to cast to MXFP4 with optional transpose.
 *
 *  MXFP4: E8M0 block scales, block size 32, native FP4 output.
 *  Adapted from quantize_transpose_nvfp4.cuh and mxfp4_transpose.cuh.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_MXFP4_CUH_

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
namespace quantize_transpose_kernel {

using namespace ptx;
using namespace core;

// MXFP4: block size 32
constexpr size_t SCALE_DIM = 32;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_NUM = 128;

constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;  // 4
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;  // 4

constexpr size_t TILE_DIM_Y = 32;
constexpr size_t TILE_DIM_X = 128;

constexpr size_t SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;  // 1
constexpr size_t SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;  // 4

constexpr size_t TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;   // 4
constexpr size_t TILES_X = CHUNK_DIM_X / TILE_DIM_X;   // 1
constexpr size_t STAGES = TILES_Y * TILES_X;            // 4

constexpr size_t BUFFS_NUM = 2;
constexpr size_t BUFF_DIM_Y = TILE_DIM_Y;
constexpr size_t BUFF_DIM_X = TILE_DIM_X;
constexpr size_t BUFF_SIZE = BUFF_DIM_Y * BUFF_DIM_X;
constexpr size_t BUFF_SIZE_TOTAL = BUFF_SIZE * BUFFS_NUM;

constexpr size_t BUFF_IN_DIM_X = BUFF_DIM_X;
constexpr size_t BUFF_IN_SIZE = BUFF_DIM_Y * BUFF_IN_DIM_X;

// FP4: 0.5 byte per element
constexpr size_t BUFF_OUT_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_OUT_DIM_X = (BUFF_DIM_X * 4) / 8;
constexpr size_t BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

constexpr size_t BUFF_OUT_T_DIM_Y = BUFF_DIM_X;
constexpr size_t BUFF_OUT_T_DIM_X = (BUFF_DIM_Y * 4) / 8;
constexpr size_t BUFF_OUT_T_SIZE = BUFF_OUT_T_DIM_Y * BUFF_OUT_T_DIM_X;

constexpr size_t PACK_SIZE = 8;
constexpr size_t WAVES = SCALE_DIM / PACK_SIZE;  // 4

constexpr size_t SCALING_FACTORS_PER_TILE_X = TILE_DIM_X / SCALE_DIM;  // 4
constexpr size_t THREADS_X_ROWWISE = SCALING_FACTORS_PER_TILE_X;       // 4
constexpr size_t THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;  // 32

constexpr size_t ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;    // 1
constexpr size_t ITERATIONS_TRANSPOSE = BUFF_IN_DIM_X / SCALE_DIM;      // 4 (was BUFF_IN_DIM_Y for NVFP4)
constexpr size_t BUFF_OUT_IT_OFFSET = BUFF_OUT_T_DIM_X / ITERATIONS_TRANSPOSE;

// Number of 4-bit elements that span 32 banks
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM;  // 8

static_assert(BUFF_DIM_Y >= SCALE_DIM, "Buffer rows must be >= block size");
static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE, "Buffer rows must be >= rowwise thread count");

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING,
          bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
__global__ void __launch_bounds__(THREADS_NUM)
    quantize_transpose_mxfp4_kernel(
        const __grid_constant__ CUtensorMap tensor_map_input,
        const __grid_constant__ CUtensorMap tensor_map_output,
        const __grid_constant__ CUtensorMap tensor_map_output_t,
        mxfp4_scale_t *const scales_ptr,
        mxfp4_scale_t *const scales_t_ptr,
        const float *noop,
        const size_t rows,
        const size_t cols,
        const size_t scale_stride,
        const size_t scale_stride_t,
        const size_t *rng_state) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT =
      (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

  using IType2 = typename ptx::FPx2<IType>;

  if constexpr (!COMPUTE_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }

  // RNG setup
  const size_t rng_sequence =
      threadIdx.x + blockIdx.x * THREADS_NUM + blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG rng(rng_seed, rng_sequence, rng_offset);
  curanddx::uniform_bits dist;
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? dist.generate4(rng) : uint4{0, 0, 0, 0};
  int rnd_idx = 0;

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;

  const size_t chunk_rows = rows - block_offset_Y;

  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;
  const size_t scales_block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_X_t = blockIdx.y * SCALES_PER_CHUNK_Y;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_X_colwise = threadIdx.x;
  const size_t tid_Y_t = tid_X_colwise;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_t = scales_block_offset_Y_t + tid_Y_t;
  const size_t scales_offset_X_t = scales_block_offset_X_t;

  const size_t SFs_per_row = cols / SCALE_DIM;

  const bool rowwise_scale_is_within_bounds_X = (scales_offset_X_rowwise < SFs_per_row);
  const bool colwise_scale_is_within_bounds_Y = (scales_offset_Y_t < cols);

  const bool col_out_of_bounds_colwise = (block_offset_X + thread_offset_X_colwise >= cols);

  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  // Shared memory layout
  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

  constexpr size_t in_mem = buff_size_aligned_in;
  constexpr size_t out_mem_rowwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_rowwise_scales = 0;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);
  mxfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<mxfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);

  IType *cached_act_sh = in_sh;

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;
  const bool is_master_thread = (threadIdx.x == 0);

  float thread_amax = 0.0f;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input,
                    block_offset_X, block_offset_Y,
                    shmem_buff_size, &mbar[0], is_master_thread);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    if (next_stage < STAGES) {
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t next_buff_offset = next_buff * BUFF_IN_SIZE;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input,
                        global_offset_X, global_offset_Y,
                        shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;

    // COLWISE scaling (transpose path)
    if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
        const size_t in_thread_offset_Y = it * SCALE_DIM;
        const size_t in_thread_offset_X = thread_offset_X_colwise;

        const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
        const size_t out_t_thread_offset_X = it * BUFF_OUT_IT_OFFSET;

        const size_t shmem_offset_base_colwise_in =
            buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
        const size_t shmem_offset_base_colwise_out_t =
            buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

        block_amax = 0.0f;
        float in_compute_colwise[SCALE_DIM];
        IType in_colwise_IType[SCALE_DIM];

        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            in_colwise_IType[i] = in_sh[shmem_offset];
            block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
          }
          block_amax = static_cast<float>(block_amax_f16);
        } else {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            float elt = static_cast<float>(in_sh[shmem_offset]);
            if constexpr (COMPUTE_ACTIVATIONS) {
              elt = OP(elt, {});
            }
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            if constexpr (IS_CACHED_ACT_OP) {
              cached_act_sh[shmem_offset] = static_cast<IType>(elt);
            }
            if constexpr (COMPUTE_ACTIVATIONS) {
              const bool row_oob = (block_offset_Y + stage_offset_Y + in_thread_offset_Y + i >= rows);
              if (!col_out_of_bounds_colwise && !row_oob) {
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
            } else {
              block_amax = fmaxf(block_amax, fabsf(elt));
            }
            in_compute_colwise[i] = elt;
          }
        }

        // Compute E8M0 scale
        mxfp4_scale_t S_b_e8m0;
        float block_scale_inverse;

        if constexpr (ENCODE_CENTRIC) {
          mxfp4_scale_t mult_bits = compute_encoding_scaling_factor(block_amax);
          block_scale_inverse = exp2f_e8m0(mult_bits);
          int flipped = 254 - (int)mult_bits;
          S_b_e8m0 = static_cast<mxfp4_scale_t>(max(0, min(255, flipped)));
        } else {
          S_b_e8m0 = compute_decoding_scaling_factor(block_amax);
          block_scale_inverse = exp2f_rcp_e8m0(S_b_e8m0);
        }

        const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_b_e8m0;

        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

        // Quantize to FP4 and transpose
        fp4e2m1x4 regs[SCALE_DIM / 4];
#pragma unroll
        for (int e = 0; e < SCALE_DIM / 4; ++e) {
          const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            const uint64_t elts =
                *reinterpret_cast<uint64_t *>(&in_colwise_IType[4 * e]);
            regs[e] = mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                elts, block_scale_inverse_2x, rbits);
          } else {
            const float2 in01 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
            const float2 in23 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
            regs[e] = mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                in01, in23, block_scale_inverse_2x, rbits);
          }
        }

        // Swizzle and store transposed data
        const int group = thread_lane / 16;
        uint32_t val[4];
        uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

        switch (group) {
          case 0:
            val[0] = regs_4x[0]; val[1] = regs_4x[1];
            val[2] = regs_4x[2]; val[3] = regs_4x[3];
            break;
          case 1:
            val[0] = regs_4x[1]; val[1] = regs_4x[0];
            val[2] = regs_4x[3]; val[3] = regs_4x[2];
            break;
        }

        uint32_t *out_t_as_u32 =
            reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);
        out_t_as_u32[group]       = val[0];
        out_t_as_u32[(group ^ 1)] = val[1];
        out_t_as_u32[group + 2]   = val[2];
        out_t_as_u32[(group ^ 1) + 2] = val[3];
      }
    }  // RETURN_TRANSPOSE

    // ROWWISE scaling
    {
      const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t it_thread_offset_Y =
            thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const size_t shmem_offset_base_in =
            buff_offset_in + it_thread_offset_Y * BUFF_IN_DIM_X;
        const size_t shmem_offset_base_out =
            buff_offset_out + it_thread_offset_Y * BUFF_OUT_DIM_X;

        const size_t it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;
        float in_compute_rowwise[SCALE_DIM];
        Vec<IType, PACK_SIZE> in_cached[WAVES];
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        // Find AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset = shmem_offset_base_in + swizzled_thread_idx;
            in_IType[w].load_from(&in_sh[shmem_offset]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
            }
          }
          block_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
        } else if constexpr (IS_CACHED_ACT_OP) {
          __syncthreads();
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset = shmem_offset_base_in + swizzled_thread_idx;
            const bool row_oob = (row_base_rowwise + it_offset_Y >= rows);
            const bool col_oob = (block_offset_X + swizzled_thread_idx >= cols);
            const bool oob = row_oob || col_oob;
            in_cached[w].load_from(&cached_act_sh[shmem_offset]);
            if (!oob) {
              if constexpr (std::is_same_v<IType, float>) {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; ++e) {
                  block_amax = fmaxf(block_amax, fabsf(in_cached[w].data.elt[e]));
                }
              } else {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; e += 2) {
                  const IType2 in_cached_2x = {in_cached[w].data.elt[e],
                                                in_cached[w].data.elt[e + 1]};
                  ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_cached_2x);
                }
              }
            }
          }
          if constexpr (!std::is_same_v<IType, float>) {
            block_amax =
                static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
          }
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset = shmem_offset_base_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            in.load_from(&in_sh[shmem_offset]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE; ++e) {
              const size_t j = w * PACK_SIZE + e;
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
              in_compute_rowwise[j] = elt;
            }
          }
        }

        // Compute E8M0 scale
        mxfp4_scale_t S_b_e8m0;
        float block_scale_inverse;

        if constexpr (ENCODE_CENTRIC) {
          mxfp4_scale_t mult_bits = compute_encoding_scaling_factor(block_amax);
          block_scale_inverse = exp2f_e8m0(mult_bits);
          int flipped = 254 - (int)mult_bits;
          S_b_e8m0 = static_cast<mxfp4_scale_t>(max(0, min(255, flipped)));
        } else {
          S_b_e8m0 = compute_decoding_scaling_factor(block_amax);
          block_scale_inverse = exp2f_rcp_e8m0(S_b_e8m0);
        }

        // Store scale
        const size_t scales_offset_Y =
            scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
        const size_t scales_offset_X = scales_offset_X_rowwise;
        const size_t scale_idx_global =
            scales_offset_Y * scale_stride + scales_offset_X;

        const bool rowwise_scale_is_within_bounds_Y =
            (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;

        if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_b_e8m0;
        }

        // Quantize
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 4; ++e) {
            const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              const uint64_t elts =
                  *reinterpret_cast<uint64_t *>(&in_IType[w].data.elt[2 * e]);
              out.data.elt[e] =
                  mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                      elts, block_scale_inverse_2x, rbits);
            } else if constexpr (IS_CACHED_ACT_OP) {
              const uint64_t elts =
                  *reinterpret_cast<uint64_t *>(&in_cached[w].data.elt[4 * e]);
              out.data.elt[e] =
                  mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                      elts, block_scale_inverse_2x, rbits);
            } else {
              const int j = w * PACK_SIZE + 4 * e;
              const float2 in01 =
                  make_float2(in_compute_rowwise[j], in_compute_rowwise[j + 1]);
              const float2 in23 =
                  make_float2(in_compute_rowwise[j + 2], in_compute_rowwise[j + 3]);
              out.data.elt[e] =
                  mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                      in01, in23, block_scale_inverse_2x, rbits);
            }
          }
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset = shmem_offset_base_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset]);
        }
      }
    }

    __builtin_assume(thread_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    if (is_master_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output),
          global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_offset_out]));

      if constexpr (RETURN_TRANSPOSE) {
        const size_t global_offset_Y_t = block_offset_Y_t;
        const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t),
            global_offset_X_t, global_offset_Y_t,
            reinterpret_cast<uint64_t *>(&out_t_data_sh[buff_offset_out_t]));
      }

      ptx::cp_async_bulk_commit_group();
    }
  }  // stage loop

  // Store colwise scales
  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
    using ScalesVec = Vec<mxfp4_scale_t, SCALES_PER_CHUNK_Y>;
    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
    ScalesVec &scales_vec =
        *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);
    const size_t scale_idx_global =
        scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
    const size_t count =
        (chunk_rows >= CHUNK_DIM_Y) ? SCALES_PER_CHUNK_Y : (chunk_rows / SCALE_DIM);
    mxfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
    constexpr size_t vec_bytes = SCALES_PER_CHUNK_Y * sizeof(mxfp4_scale_t);
    if (count == SCALES_PER_CHUNK_Y &&
        (reinterpret_cast<uintptr_t>(dst) % vec_bytes == 0)) {
      scales_vec.store_to(dst);
    } else {
      scales_vec.store_to_elts(dst, 0, count);
    }
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

}  // namespace quantize_transpose_kernel

/// Host wrapper for MXFP4 quantize + transpose.
template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_transpose(const Tensor &input, const Tensor *noop, Tensor *output,
                        const QuantizationConfig *quant_config,
                        cudaStream_t stream) {
  using namespace quantize_transpose_kernel;
  using namespace ptx;

  const bool use_stochastic_rounding =
      quant_config ? quant_config->stochastic_rounding : false;
  const bool return_transpose = output->has_columnwise_data();
  const bool use_encode_centric =
      quant_config ? quant_config->encode_centric : false;

  checkCuDriverContext(stream);
  if (noop) {
    CheckNoopTensor(*noop, "cast_noop");
  }
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "MXFP4 output tensor must be allocated.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");

  if (return_transpose) {
    NVTE_CHECK(output->has_columnwise_data(),
               "MXFP4 transposed output tensor must be allocated.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Transposed scaling tensor must be allocated.");
  }

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  NVTE_CHECK(rows % 32 == 0, "Number of rows must be a multiple of 32");
  NVTE_CHECK(cols % 32 == 0, "Number of cols must be a multiple of 32");

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_NUM;

  const size_t scale_stride =
      static_cast<size_t>(output->scale_inv.shape[1]);
  const size_t scale_stride_t =
      return_transpose
          ? static_cast<size_t>(output->columnwise_scale_inv.shape[1])
          : 0;

  mxfp4_scale_t *const scales_ptr =
      reinterpret_cast<mxfp4_scale_t *>(output->scale_inv.dptr);
  mxfp4_scale_t *const scales_t_ptr =
      return_transpose
          ? reinterpret_cast<mxfp4_scale_t *>(output->columnwise_scale_inv.dptr)
          : nullptr;

  const float *noop_ptr =
      (noop && noop->data.dptr) ? reinterpret_cast<const float *>(noop->data.dptr) : nullptr;

  // RNG state
  const NVTETensor rng_state_tensor =
      (quant_config != nullptr) ? quant_config->rng_state : nullptr;
  const size_t *rng_state = nullptr;
  if (rng_state_tensor != nullptr) {
    Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
    rng_state = reinterpret_cast<const size_t *>(rng_state_te_tensor.data.dptr);
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      alignas(64) CUtensorMap tensor_map_input{};
      alignas(64) CUtensorMap tensor_map_output{};
      alignas(64) CUtensorMap tensor_map_output_t{};

      create_2D_tensor_map(tensor_map_input, input.data,
                           rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                           cols, 0, sizeof(IType) * 8);

      create_2D_tensor_map(tensor_map_output, output->data,
                           rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                           cols, 0, 4);  // 4 bits

      if (return_transpose) {
        create_2D_tensor_map(tensor_map_output_t, output->columnwise_data,
                             cols, rows, BUFF_DIM_X, BUFF_DIM_Y,
                             rows, 0, 4);
      }

      // Shared memory
      constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
      constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
      constexpr size_t buff_size_aligned_in =
          DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
      constexpr size_t buff_size_aligned_out =
          DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
      constexpr size_t buff_size_scales =
          (CHUNK_DIM_Y * CHUNK_DIM_X) / 32 * sizeof(mxfp4_scale_t);

      constexpr size_t dshmem_size =
          buff_size_aligned_in + 2 * buff_size_aligned_out + buff_size_scales +
          TMA_SHMEM_ALIGNMENT + 1024;

      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          use_stochastic_rounding, USE_SR,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_transpose, RET_T,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_encode_centric, ENC_C, {
                    auto kernel =
                        &quantize_transpose_mxfp4_kernel<
                            COMPUTE_ACTIVATIONS, ParamOP, OP,
                            IType, USE_SR, RET_T, ENC_C>;

                    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(dshmem_size)));

                    kernel<<<grid, block_size, dshmem_size, stream>>>(
                        tensor_map_input, tensor_map_output, tensor_map_output_t,
                        scales_ptr, scales_t_ptr, noop_ptr,
                        rows, cols, scale_stride, scale_stride_t, rng_state);
                  })));
  );  // TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_MXFP4_CUH_
