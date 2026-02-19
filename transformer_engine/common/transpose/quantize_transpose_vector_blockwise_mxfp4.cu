/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cuda/barrier>
#include <utility>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/transpose/cast_transpose.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#include "common/util/curanddx.hpp"

// Turn this on/off to match your build configuration
#ifndef MXFP4_SIMULATE_WITH_FP8
#define MXFP4_SIMULATE_WITH_FP8 1
#endif

// Custom Switch to allow FP8 output when simulating, or FP4 output when native
#define TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT_MXFP4(dtype, type, ...)   \
  switch (dtype) {                                                      \
    case DType::kFloat4E2M1: {                                          \
      if (MXFP4_SIMULATE_WITH_FP8) {                                    \
          NVTE_ERROR("Output must be FP8 (e4m3) when simulating MXFP4");\
      }                                                                 \
      using type = unsigned char; /* Packed storage */                  \
      { __VA_ARGS__ }                                                   \
    } break;                                                            \
    case DType::kFloat8E4M3: {                                          \
      if (!MXFP4_SIMULATE_WITH_FP8) {                                   \
          NVTE_ERROR("Output must be FP4 (e2m1) when using native MXFP4");\
      }                                                                 \
      using type = fp8e4m3;                                             \
      { __VA_ARGS__ }                                                   \
    } break;                                                            \
    default:                                                            \
      NVTE_ERROR("Invalid type for MXFP4 quantization. Expected FP4 or FP8."); \
  }

namespace transformer_engine {

#if CUDA_VERSION >= 12080
namespace quantize_transpose_mxfp4 {
namespace {

using std::int32_t;
using std::uint32_t;
using std::uint8_t;

using transformer_engine::detail::TypeExtrema;

using RNG = transformer_engine::curanddx::detail::philox4x32_native_state<10>;

constexpr int kThreadsPerWarp = 32;

// ------------------------
// MXFP4 Helpers 
// ------------------------

// LUT: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
__device__ __constant__ uint8_t FP4_TO_FP8E4M3_LUT[16] = {
    // positive grid
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    // negative grid (sign bit set)
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
};

template <bool USE_SR>
__device__ __forceinline__ uint8_t encode_fp4_index_from_scaled(float z, uint32_t rbits) {
  float a = fabsf(z);
  const int sign_bit = (z < 0.0f) ? 1 : 0;

  int lo_idx = 0, hi_idx = 0;
  float lo_val = 0.0f, hi_val = 0.0f;

  if (a <= 0.0f) {
    lo_idx = hi_idx = 0; lo_val = hi_val = 0.0f;
  } else if (a < 0.5f) {
    lo_idx = 0; hi_idx = 1; lo_val = 0.0f; hi_val = 0.5f;
  } else if (a < 1.0f) {
    lo_idx = 1; hi_idx = 2; lo_val = 0.5f; hi_val = 1.0f;
  } else if (a < 1.5f) {
    lo_idx = 2; hi_idx = 3; lo_val = 1.0f; hi_val = 1.5f;
  } else if (a < 2.0f) {
    lo_idx = 3; hi_idx = 4; lo_val = 1.5f; hi_val = 2.0f;
  } else if (a < 3.0f) {
    lo_idx = 4; hi_idx = 5; lo_val = 2.0f; hi_val = 3.0f;
  } else if (a < 4.0f) {
    lo_idx = 5; hi_idx = 6; lo_val = 3.0f; hi_val = 4.0f;
  } else if (a < 6.0f) {
    lo_idx = 6; hi_idx = 7; lo_val = 4.0f; hi_val = 6.0f;
  } else {
    lo_idx = hi_idx = 7; lo_val = hi_val = 6.0f;
  }

  int mag_idx;
  if (lo_idx == hi_idx) {
    mag_idx = lo_idx;
  } else if constexpr (USE_SR) {
    float p = (a - lo_val) / (hi_val - lo_val);
    float u = (rbits >> 8) * (1.0f / 16777216.0f);
    mag_idx = (u < p) ? hi_idx : lo_idx;
  } else {
    float mid = 0.5f * (lo_val + hi_val);
    if (a == mid) {
      mag_idx = (lo_idx % 2 == 0) ? lo_idx : hi_idx;
    } else {
      mag_idx = (a > mid) ? hi_idx : lo_idx;
    }
  }

  mag_idx &= 0x7;
  return static_cast<uint8_t>(mag_idx | (sign_bit << 3));
}

// ------------------------
// Scale Logic
// ------------------------

template <typename ScaleType>
__device__ __forceinline__ ScaleType ComputeDecodeScaleFP4(const float amax,
                                                           const float global_encode_scale) {
  constexpr float fp4_max = 6.0f; 
  float effective_amax = (amax * global_encode_scale) / fp4_max;
  
  if (effective_amax <= 1.0e-9f) {
      return static_cast<ScaleType>(0);
  }

  float exponent_f = roundf(log2f(effective_amax));
  int exponent_i = (int)exponent_f;

  exponent_i = max(-127, min(128, exponent_i));
  return static_cast<ScaleType>(exponent_i + 127);
}

template <typename ScaleType>
__device__ __forceinline__ float ComputeEncodeScaleFP4(ScaleType decode_scale,
                                                       const float global_encode_scale) {
  int exponent = (int)decode_scale - 127;
  float local_scale = exp2f((float)(-exponent));
  return global_encode_scale * local_scale;
}

__device__ __forceinline__ float ComputeGlobalEncodeScaleFP4(const float global_amax) {
  constexpr float fp4_max = 6.0f;
  if (global_amax <= 1.0e-9f) {
    return 1.0f;
  }
  return fp4_max / global_amax;
}

// ------------------------
// Kernels
// ------------------------

constexpr int kTileDim = 128;
constexpr int kNVecIn = 8;             
constexpr int kNVecOut = 16;           
constexpr int kNVecSMem = 2;           
constexpr int kThreadsPerBlock = 256;  

constexpr int kSMemRow = kTileDim;
constexpr int kSMemCol = (kTileDim / kNVecSMem) + 1;
constexpr int kSMemSize = kSMemRow * kSMemCol * kNVecSMem;
constexpr int kNumThreadsLoad = kTileDim / kNVecIn;    
constexpr int kNumThreadsStore = kTileDim / kNVecOut;  

static __device__ constexpr unsigned int WARP_REDUCE_AMAX_GROUP_MASKS[8] = {
    0x01010101, 0x02020202, 0x04040404, 0x08080808, 0x10101010, 0x20202020, 0x40404040, 0x80808080};

template <int group_size, int shfl_down_stride>
__device__ __forceinline__ float groupMax(float val, unsigned int groupMask) {
  for (int offset = group_size / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(groupMask, val, offset * shfl_down_stride));
  }
  return val;
}

__device__ __forceinline__ uint32_t get_rbits(RNG& rng, uint4& random_uint4, int& rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    random_uint4 = rng.generate4();
  }
  const uint32_t* const rbits_arr = reinterpret_cast<uint32_t*>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}

template <class ScaleType>
__device__ __forceinline__ size_t scale_factor_swizzled_offset(size_t row_idx, size_t col_idx,
                                                               uint32_t col_length) {
  constexpr uint32_t kTotalRowsPerBaseBlock = 128;
  constexpr uint32_t kRowsPerBaseBlockCol = 32;
  constexpr uint32_t kColsPerBaseBlockCol = 4;

  const size_t rb = row_idx / kTotalRowsPerBaseBlock;
  const size_t rem = row_idx % kTotalRowsPerBaseBlock;
  const size_t d4 = rem / kRowsPerBaseBlockCol;
  const size_t d3 = rem % kRowsPerBaseBlockCol;
  const size_t cbg = col_idx / kColsPerBaseBlockCol;
  const size_t d5 = col_idx % kColsPerBaseBlockCol;

  const size_t cbg_cnt = DIVUP(col_length, kColsPerBaseBlockCol);
  return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 + d4 * kColsPerBaseBlockCol + d5;
}

// Native FP4 PTX
__device__ __forceinline__ __nv_fp4x4_e2m1 cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const uint32_t rbits) {
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    uint16_t out_4x;
    asm volatile(
        "{\n"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {%3, %4, %1, %2}, %5; \n\t"
        "}"
        : "=h"(out_4x)
        : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x), "r"(rbits));
    return *reinterpret_cast<__nv_fp4x4_e2m1*>(&out_4x);
  } else {
    NVTE_DEVICE_ERROR("FP4 cvt.rs PTX error.");
    uint16_t dummy = 0;
    return *reinterpret_cast<__nv_fp4x4_e2m1*>(&dummy);
  }
}

__device__ __forceinline__ __nv_fp4x4_e2m1 cvt_fp32_to_fp4_4x_with_rn(const float2 in01,
                                                                      const float2 in23,
                                                                      const uint32_t rbits) {
  constexpr bool has_fp4 = ARCH_BLACKWELL_FAMILY;
  if constexpr (has_fp4) {
    uint32_t out_4x;  
    asm volatile(
        "{\n"
        ".reg.b8 f0; \n\t"
        ".reg.b8 f1; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, %1, %2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, %3, %4;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}"
        : "=r"(out_4x)
        : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x));
    return reinterpret_cast<__nv_fp4x4_e2m1*>(&out_4x)[0];
  } else {
    NVTE_DEVICE_ERROR("FP4 cvt PTX error.");
    uint16_t dummy = 0;
    return *reinterpret_cast<__nv_fp4x4_e2m1*>(&dummy);
  }
}

template <bool kApplyStochasticRounding>
__device__ __forceinline__ __nv_fp4x4_e2m1 cvt_fp32_to_fp4_4x(const float2 in01, const float2 in23,
                                                              const uint32_t rbits) {
  if constexpr (kApplyStochasticRounding) {
    return cvt_fp32_to_fp4_4x_with_stochastic_rounding(in01, in23, rbits);
  } else {
    return cvt_fp32_to_fp4_4x_with_rn(in01, in23, rbits);
  }
}

template <bool kReturnIdentity, bool kReturnTranspose, bool kIsE8Scaling, bool kAligned,
          typename CType, typename IType, typename OType, typename ScaleType, bool kSwizzledScale,
          bool kApplyStochasticRounding, bool kIs2DBlockScaling>
__global__ void __launch_bounds__(kThreadsPerBlock) block_scaled_1d_cast_transpose_kernel(
    const IType* const input, const float* global_amax, OType* const output_c,
    OType* const output_t, ScaleType* const tile_scales_inv_c, ScaleType* const tile_scales_inv_t,
    const size_t row_length, const size_t num_rows, const size_t scale_stride_x,
    const size_t scale_stride_y, const size_t scale_t_stride_x, const size_t scale_t_stride_y,
    const size_t kScaleBlockDim, const float epsilon, const size_t* rng_state,
    const float* noop_ptr) {
  
  // ======================================================================================
  // Simulation vs Native Logic
  // ======================================================================================
  #if MXFP4_SIMULATE_WITH_FP8
     // SIMULATION: 1 element per byte (unpacked)
     constexpr int kNFP4PerContainer = 1; 
  #else
     // NATIVE: 2 elements per byte (packed)
     constexpr int kNFP4PerContainer = 2; 
  #endif
  
  constexpr int kNVecContainer = kNVecOut / kNFP4PerContainer;
  
  using SMemVec = Vec<IType, kNVecSMem>;
  using OVec = Vec<OType, kNVecContainer>;
  union IVec {
    Vec<IType, kNVecIn> input_type;
    Vec<SMemVec, kNVecIn / kNVecSMem> smem_type;
  };

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) return;

  const size_t block_idx_x = blockIdx.x;
  const size_t block_idx_y = blockIdx.y;
  const size_t rng_sequence =
      threadIdx.x + block_idx_x * kThreadsPerBlock + block_idx_y * gridDim.x * kThreadsPerBlock;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG rng;
  rng.init(rng_seed, rng_sequence, rng_offset);
  uint4 random_uint4 = kApplyStochasticRounding ? rng.generate4() : uint4{0, 0, 0, 0};
  int rnd_idx = 0;

  extern __shared__ char smem_base[];
  SMemVec* smem = reinterpret_cast<SMemVec*>(&smem_base[0]);


  constexpr int kFP4BlockScalingSize = 32;
  constexpr int k2DBlockAmaxRowDim = kIs2DBlockScaling ? (kTileDim / kFP4BlockScalingSize) : 1;
  constexpr int k2DBlockAmaxColDim = kIs2DBlockScaling ? kNumThreadsStore : 1;

  constexpr int kNumRowsPerWarp = kThreadsPerWarp / kNumThreadsStore;  // 4
  constexpr int k2DBlockAmaxReduceDim =
      kIs2DBlockScaling ? (kFP4BlockScalingSize / kNumRowsPerWarp) : 1;
  __shared__ CType amax_smem_red[k2DBlockAmaxRowDim][k2DBlockAmaxColDim][k2DBlockAmaxReduceDim];
  __shared__ CType amax_smem[k2DBlockAmaxRowDim][k2DBlockAmaxColDim];

  // Step 1: Load input to shared memory
  {
    constexpr int r_stride = kThreadsPerBlock / kNumThreadsLoad;
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s = (threadIdx.x % kNumThreadsLoad) * (kNVecIn / kNVecSMem);
    int r_s = threadIdx.x / kNumThreadsLoad;
    const size_t c_g = block_idx_x * kTileDim + c_s * kNVecSMem;
    size_t r_g = block_idx_y * kTileDim + r_s;
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;
    const size_t num_ele = (c_g < row_length ? min(static_cast<size_t>(kNVecIn), row_length - c_g) : 0);
    const IType* input_g = &input[r_g * row_length + c_g];
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      IVec input_vec;
      if constexpr (kAligned) {
        input_vec.input_type.load_from(input_g);
      } else {
        if (r_g < num_rows) {
          input_vec.input_type.load_from_elts(input_g, 0, num_ele);
        } else {
          input_vec.input_type.clear();
        }
      }
#pragma unroll
      for (int i = 0; i < kNVecIn / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        smem[r * kSMemCol + c] = input_vec.smem_type.data.elt[i];
      }
      input_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) { r_g += r_stride; }
    }
  }

  __syncthreads();

  const int kNumThreadsReduce = kScaleBlockDim / kNVecOut;
  const float global_encode_scale =
      (global_amax == nullptr) ? 1.0f : ComputeGlobalEncodeScaleFP4(global_amax[0]);
  const float global_decode_scale = 1.0 / global_encode_scale;

  // Step 2: Compute Scale Factors and Cast (IDENTITY)
  if constexpr (kReturnIdentity || kIs2DBlockScaling) {
    constexpr int r_stride = kThreadsPerBlock / kNumThreadsStore;
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s = (threadIdx.x % kNumThreadsStore) * (kNVecOut / kNVecSMem);
    int r_s = threadIdx.x / kNumThreadsStore;
    const size_t c_g = block_idx_x * kTileDim + c_s * kNVecSMem;
    size_t r_g = block_idx_y * kTileDim + r_s;
    const size_t stride_g = static_cast<size_t>(r_stride) * row_length;
    const size_t num_ele = (c_g < row_length ? min(static_cast<size_t>(kNVecOut / kNFP4PerContainer),
                                (row_length - c_g) / kNFP4PerContainer) : 0);
    OType* output_g = &output_c[(r_g * row_length + c_g) / kNFP4PerContainer];
    
    const unsigned src_lane = (threadIdx.x % kThreadsPerWarp) / kNumThreadsReduce * kNumThreadsReduce;
    const unsigned mask = ((1 << kNumThreadsReduce) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsReduce) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut / kNVecSMem];
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
      CType amax = 0;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[j]));
        }
      }
      if constexpr (kIsE8Scaling) {
#pragma unroll
        for (int delta = kNumThreadsReduce / 2; delta > 0; delta /= 2) {
          const float other_amax = __shfl_down_sync(mask, amax, delta);
          __builtin_assume(amax >= 0); __builtin_assume(other_amax >= 0);
          amax = fmaxf(amax, other_amax);
        }
        amax = __shfl_sync(mask, amax, src_lane);
      }
      if constexpr (kIs2DBlockScaling) {
         constexpr int kNumRowsPerIter = kThreadsPerBlock / kNumThreadsStore;
         int warp_idx = threadIdx.x / kThreadsPerWarp;
         int tid_in_warp_x = threadIdx.x % kNumThreadsStore;
         int tid_in_warp_y = (threadIdx.x / kNumThreadsStore) % kNumRowsPerWarp;
         CType amax_warp_reduced = groupMax<kNumRowsPerWarp, kNumThreadsStore>(
            amax, WARP_REDUCE_AMAX_GROUP_MASKS[tid_in_warp_x]);
         int data_row_idx = iter * kNumRowsPerIter + warp_idx * kNumRowsPerWarp + tid_in_warp_y;
         if (tid_in_warp_y == 0) {
          amax_smem_red[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x]
                       [warp_idx % k2DBlockAmaxReduceDim] = amax_warp_reduced;
         }
         __syncthreads();
         if (data_row_idx % kFP4BlockScalingSize == 0) {
          CType amax_2d = 0.0;
          for (int i = 0; i < k2DBlockAmaxReduceDim; i++) {
            amax_2d = fmaxf(amax_2d,
                            amax_smem_red[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x][i]);
          }
          amax_smem[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x] = amax_2d;
         }
         __syncthreads();
         amax = amax_smem[data_row_idx / kFP4BlockScalingSize][tid_in_warp_x];
      }

      if constexpr (kReturnIdentity) {
        ScaleType scale_inv = ComputeDecodeScaleFP4<ScaleType>(amax, global_encode_scale);
        float encode_scale = ComputeEncodeScaleFP4<ScaleType>(scale_inv, global_encode_scale);

      bool write_scale_inv = is_src_lane;
      if constexpr (!kAligned) {
        write_scale_inv &= (r_g < num_rows);
        write_scale_inv &= (c_g < row_length);
      }
      if (write_scale_inv) {
        size_t row_idx = block_idx_y * kTileDim + r_s;
        size_t col_idx = block_idx_x * (kNumThreadsStore / kNumThreadsReduce) +
                         (threadIdx.x % kNumThreadsStore) / kNumThreadsReduce;
        if constexpr (kSwizzledScale) {
          size_t offset = scale_factor_swizzled_offset<ScaleType>(
              row_idx, col_idx, DIVUP(row_length, kScaleBlockDim));
          tile_scales_inv_c[offset] = scale_inv;
        } else {
          tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
        }
      }
      
      OVec output_vec;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; i += 2) {
        float vals[4];
        vals[0] = static_cast<float>(smem_vec[i].data.elt[0]) * encode_scale;
        vals[1] = static_cast<float>(smem_vec[i].data.elt[1]) * encode_scale;
        vals[2] = static_cast<float>(smem_vec[i + 1].data.elt[0]) * encode_scale;
        vals[3] = static_cast<float>(smem_vec[i + 1].data.elt[1]) * encode_scale;

        const uint32_t rbits = kApplyStochasticRounding ? get_rbits(rng, random_uint4, rnd_idx) : 0;
        const uint32_t rbits2 = kApplyStochasticRounding ? get_rbits(rng, random_uint4, rnd_idx) : 0; 

        #if MXFP4_SIMULATE_WITH_FP8
          // SIMULATION (Identity): Packed as 4 individual bytes.
          // i steps by 2. i*2 maps to 4 indices. Correct.
          uint8_t idx0 = encode_fp4_index_from_scaled<kApplyStochasticRounding>(vals[0], rbits);
          uint8_t idx1 = encode_fp4_index_from_scaled<kApplyStochasticRounding>(vals[1], rbits);
          uint8_t idx2 = encode_fp4_index_from_scaled<kApplyStochasticRounding>(vals[2], rbits2);
          uint8_t idx3 = encode_fp4_index_from_scaled<kApplyStochasticRounding>(vals[3], rbits2);

          output_vec.data.elt[i * 2 + 0] = reinterpret_cast<const OType&>(FP4_TO_FP8E4M3_LUT[idx0]);
          output_vec.data.elt[i * 2 + 1] = reinterpret_cast<const OType&>(FP4_TO_FP8E4M3_LUT[idx1]);
          output_vec.data.elt[i * 2 + 2] = reinterpret_cast<const OType&>(FP4_TO_FP8E4M3_LUT[idx2]);
          output_vec.data.elt[i * 2 + 3] = reinterpret_cast<const OType&>(FP4_TO_FP8E4M3_LUT[idx3]);

        #else
          // NATIVE: Packed as 2 bytes.
          __nv_fp4x4_e2m1 out_4x = cvt_fp32_to_fp4_4x<kApplyStochasticRounding>(
              {vals[0], vals[1]}, {vals[2], vals[3]}, rbits);
          output_vec.data.elt[i] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[0];
          output_vec.data.elt[i + 1] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[1];
        #endif
      }

      if constexpr (kAligned) {
        output_vec.store_to(output_g);
      } else {
        if (r_g < num_rows) {
          output_vec.store_to_elts(output_g, 0, num_ele);
        }
        }
        output_g += stride_g / kNFP4PerContainer;
      }
      r_s += r_stride;
      if constexpr (!kAligned) { r_g += r_stride; }
    }
  }

  __syncthreads();

  // Step 3: Transpose
  if constexpr (kReturnTranspose) {
    constexpr int c_stride = kThreadsPerBlock / kNumThreadsStore;
    constexpr int num_iterations = kTileDim / (c_stride * kNVecSMem);
    const int r_s = (threadIdx.x % kNumThreadsStore) * kNVecOut;
    int c_s = threadIdx.x / kNumThreadsStore;
    size_t r_g = block_idx_x * kTileDim + c_s * kNVecSMem;
    const size_t c_g = block_idx_y * kTileDim + r_s;
    const size_t stride_g = static_cast<size_t>(c_stride) * kNVecSMem * num_rows;
    const size_t num_ele = (c_g < num_rows ? min(static_cast<size_t>(kNVecOut / kNFP4PerContainer),
                                                 (num_rows - c_g) / kNFP4PerContainer) : 0);
    OType* output_g = &output_t[(r_g * num_rows + c_g) / kNFP4PerContainer];
    const unsigned src_lane = (threadIdx.x % kThreadsPerWarp) / kNumThreadsReduce * kNumThreadsReduce;
    const unsigned mask = ((1 << kNumThreadsReduce) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsReduce) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut];
#pragma unroll
      for (int i = 0; i < kNVecOut; ++i) {
        int r = r_s + i;
        int c = c_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
#pragma unroll
      for (int smem_idx = 0; smem_idx < kNVecSMem; ++smem_idx) {
        CType amax = 0;
        if constexpr (kIs2DBlockScaling) {
          int orig_row_block = (c_s * kNVecSMem + smem_idx) / kFP4BlockScalingSize;
          int orig_col_chunk = r_s / kNVecOut;
          amax = amax_smem[orig_row_block][orig_col_chunk];
        } else {
#pragma unroll
          for (int i = 0; i < kNVecOut; ++i) {
            amax = fmaxf(amax, fabsf(smem_vec[i].data.elt[smem_idx]));
          }
        }
        if constexpr (kIsE8Scaling) {
#pragma unroll
          for (int delta = kNumThreadsReduce / 2; delta > 0; delta /= 2) {
            const float other_amax = __shfl_down_sync(mask, amax, delta);
            __builtin_assume(amax >= 0); __builtin_assume(other_amax >= 0);
            amax = fmaxf(amax, other_amax);
          }
          amax = __shfl_sync(mask, amax, src_lane);
        }
        
        ScaleType scale_inv = ComputeDecodeScaleFP4<ScaleType>(amax, global_encode_scale);
        float encode_scale = ComputeEncodeScaleFP4<ScaleType>(scale_inv, global_encode_scale);

        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (r_g + smem_idx < row_length);
          write_scale_inv &= (c_g < num_rows);
        }
        if (write_scale_inv) {
          size_t row_idx = block_idx_x * kTileDim + c_s * kNVecSMem + smem_idx;
          size_t col_idx = (block_idx_y * (kNumThreadsStore / kNumThreadsReduce) +
                            (threadIdx.x % kNumThreadsStore) / kNumThreadsReduce);
          if constexpr (kSwizzledScale) {
            size_t offset = scale_factor_swizzled_offset<ScaleType>(
                row_idx, col_idx, DIVUP(num_rows, kScaleBlockDim));
            tile_scales_inv_t[offset] = scale_inv;
          } else {
            tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
          }
        }
        OVec output_vec;
#pragma unroll
        for (int i = 0; i < kNVecOut / kNFP4PerContainer; i += 2) {
          float vals[4];
          #if MXFP4_SIMULATE_WITH_FP8
             // SIMULATION (Transpose): Unpacked.
             // Transpose processes 1 scalar per vector (via smem_idx).
             // We access 2 inputs: smem_vec[i] and smem_vec[i+1].
             // Output indices: i and i+1. (0,1 .. 14,15).
             
             vals[0] = static_cast<float>(smem_vec[i].data.elt[smem_idx]) * encode_scale;
             vals[1] = static_cast<float>(smem_vec[i+1].data.elt[smem_idx]) * encode_scale;

             const uint32_t rbits = kApplyStochasticRounding ? get_rbits(rng, random_uint4, rnd_idx) : 0;
             uint8_t idx0 = encode_fp4_index_from_scaled<kApplyStochasticRounding>(vals[0], rbits);
             uint8_t idx1 = encode_fp4_index_from_scaled<kApplyStochasticRounding>(vals[1], rbits);
             
             // [CORRECTED INDEXING: Use linear i, i+1]
             output_vec.data.elt[i]   = reinterpret_cast<const OType&>(FP4_TO_FP8E4M3_LUT[idx0]);
             output_vec.data.elt[i+1] = reinterpret_cast<const OType&>(FP4_TO_FP8E4M3_LUT[idx1]);

          #else
             // NATIVE: Packed (2 elements per byte)
             vals[0] = static_cast<float>(smem_vec[2 * i].data.elt[smem_idx]) * encode_scale;
             vals[1] = static_cast<float>(smem_vec[2 * i + 1].data.elt[smem_idx]) * encode_scale;
             vals[2] = static_cast<float>(smem_vec[2 * (i + 1)].data.elt[smem_idx]) * encode_scale;
             vals[3] = static_cast<float>(smem_vec[2 * (i + 1) + 1].data.elt[smem_idx]) * encode_scale;
             
             const uint32_t rbits = kApplyStochasticRounding ? get_rbits(rng, random_uint4, rnd_idx) : 0;
             __nv_fp4x4_e2m1 out_4x = cvt_fp32_to_fp4_4x<kApplyStochasticRounding>({vals[0],vals[1]}, {vals[2],vals[3]}, rbits);
             output_vec.data.elt[i] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[0];
             output_vec.data.elt[i + 1] = reinterpret_cast<__nv_fp4x2_storage_t*>(&out_4x)[1];
          #endif
        }
        if constexpr (kAligned) {
          output_vec.store_to(output_g + smem_idx * num_rows / kNFP4PerContainer);
        } else {
          if (r_g + smem_idx < row_length) {
            output_vec.store_to_elts(output_g + smem_idx * num_rows / kNFP4PerContainer, 0, num_ele);
          }
        }
      }
      output_g += stride_g / kNFP4PerContainer;
      c_s += c_stride;
      if constexpr (!kAligned) { r_g += c_stride * kNVecSMem; }
    }
  }
}

}  // namespace
}  // namespace quantize_transpose_mxfp4
#endif  // CUDA_VERSION >= 12080

namespace detail {

void quantize_transpose_vector_blockwise_mxfp4(
    const SimpleTensor& input, const SimpleTensor& global_amax, SimpleTensor& scale_inv,
    SimpleTensor& scale_inv_t, SimpleTensor& output, SimpleTensor& output_t, const float epsilon,
    const bool return_identity, const bool return_transpose, const bool pow2_scale,
    const bool swizzled_scale, const bool use_stochastic_rounding,
    const NVTETensor rng_state_tensor, const bool use_2d_quantization,
    const SimpleTensor& noop_tensor, cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_vector_blockwise_mxfp4);
#if CUDA_VERSION >= 12080

  // [MXFP4] Validation
  if (!return_identity && !return_transpose) return;

  // [MXFP4] Use MXFP4 Namespace
  using namespace transformer_engine::quantize_transpose_mxfp4;

  const size_t row_length = input.shape.size() > 0 ? input.shape.at(input.shape.size() - 1) : 1u;
  size_t num_elements = row_length;
  size_t num_rows = 1;
  for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
    num_rows *= input.shape.at(i);
    num_elements *= input.shape.at(i);
  }
  if (num_elements == 0) return;

  size_t scale_stride_x = return_identity ? 1 : 0;
  size_t scale_stride_y = return_identity ? scale_inv.shape[1] : 0;
  size_t scale_t_stride_x = return_transpose ? 1 : 0;
  size_t scale_t_stride_y = return_transpose ? scale_inv_t.shape[1] : 0;

  const size_t num_blocks_x = DIVUP(row_length, static_cast<size_t>(kTileDim));
  const size_t num_blocks_y = DIVUP(num_rows, static_cast<size_t>(kTileDim));

  const float* noop_ptr = reinterpret_cast<const float*>(noop_tensor.dptr);
  const size_t* rng_state = nullptr;
  if (rng_state_tensor != nullptr) {
    Tensor& rng = *convertNVTETensor(rng_state_tensor);
    rng_state = reinterpret_cast<const size_t*>(rng.data.dptr);
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,
      // [FIX] Use local macro checking MXFP4_SIMULATE_WITH_FP8
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT_MXFP4(
          output.dtype, OutputType,

          dim3 grid(num_blocks_x, num_blocks_y, 1);

          // [MXFP4] Params
          using ScaleType = uint8_t;
          constexpr int kScaleBlockDim = 32;
          constexpr bool kPow2Scale = true;
          constexpr bool kIsE8Scaling = true; 

          const bool full_tile = row_length % kTileDim == 0 && num_rows % kTileDim == 0;

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_identity, kReturnIdentity,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  return_transpose, kReturnTranspose,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      full_tile, kAligned,
                      TRANSFORMER_ENGINE_SWITCH_CONDITION(
                          swizzled_scale, kSwizzledScale,
                          TRANSFORMER_ENGINE_SWITCH_CONDITION(
                              use_stochastic_rounding, kApplyStochasticRounding,
                              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                                  use_2d_quantization, kIs2DBlockScaling,

                                  size_t smem_bytes = kSMemSize * sizeof(InputType);
                                  
                                  auto kernel = block_scaled_1d_cast_transpose_kernel<
                                      kReturnIdentity, kReturnTranspose, kIsE8Scaling, kAligned,
                                      float, InputType, OutputType, ScaleType, kSwizzledScale,
                                      kApplyStochasticRounding, kIs2DBlockScaling>;
                                  
                                  if (smem_bytes >= 48 * 1024) {
                                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                                  } 
                                  kernel<<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
                                      reinterpret_cast<const InputType*>(input.dptr),
                                      reinterpret_cast<const float*>(global_amax.dptr),
                                      reinterpret_cast<OutputType*>(output.dptr),
                                      reinterpret_cast<OutputType*>(output_t.dptr),
                                      reinterpret_cast<ScaleType*>(scale_inv.dptr),
                                      reinterpret_cast<ScaleType*>(scale_inv_t.dptr), row_length,
                                      num_rows, scale_stride_x, scale_stride_y, scale_t_stride_x,
                                      scale_t_stride_y, kScaleBlockDim, epsilon, rng_state,
                                      noop_ptr);) 
                              )
                          )
                      )
                  )
              )
          )
      )

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("MXFP4 support requires CUDA 12.8+");
#endif
}

}  // namespace detail
}  // namespace transformer_engine
