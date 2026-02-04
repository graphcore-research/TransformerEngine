/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file mxfp4_transpose.cuh
 * \brief CUDA kernels to cast to MXFP4 and transpose.
 */

#ifndef TRANSFORMER_ENGINE_MXFP4_TRANSPOSE_CUH_
#define TRANSFORMER_ENGINE_MXFP4_TRANSPOSE_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>

#include "../common.h"
#include "../utils.cuh"
#include "curanddx.hpp"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/transformer_engine.h"

// Turn this on when you want to drive cuBLAS MXFP8 via MXFP4 encoding.
#ifndef MXFP4_SIMULATE_WITH_FP8
#define MXFP4_SIMULATE_WITH_FP8 1
#endif

#if MXFP4_SIMULATE_WITH_FP8
# pragma message("MXFP4_SIMULATE_WITH_FP8 = 1 (FP8 simulation path)")
#else
# pragma message("MXFP4_SIMULATE_WITH_FP8 = 0 (native FP4 path)")
#endif

#ifndef MXFP4_DEBUG_PRINTS
#define MXFP4_DEBUG_PRINTS 0   // set 0 to disable
#endif



namespace transformer_engine {

#if FP4_TYPE_SUPPORTED

namespace mxfp4_transpose {

using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                     curanddx::PhiloxRounds<10>() +
                     curanddx::SM<800>() +
                     curanddx::Thread());

using namespace ptx;

// MXFP4 scales are stored as E8M0 (uint8).
using mxfp4_scale_t = e8m0_t;

// MXFP4 block size is 32 elements.
constexpr size_t SCALE_DIM = 32;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_NUM = 128;

constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;

constexpr size_t SCALES_PER_THREAD =
    2 * (CHUNK_DIM_Y * CHUNK_DIM_X) / SCALE_DIM / THREADS_NUM;
constexpr size_t RNG_GENS_PER_THREAD =
    SCALES_PER_THREAD / 4;  // Each call generates 4x uint32_t random numbers

constexpr size_t TILE_DIM_Y = 32;
constexpr size_t TILE_DIM_X = 128;

// Should this be SCALE_DIM or BLOCK_DIM? Both are 16 in NVFP4,
// but here SCALE_DIM = 32 for MXFP4.
// These are used only as “scales per tile” divisors.
constexpr size_t SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;
constexpr size_t SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;

constexpr size_t TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;
constexpr size_t TILES_X = CHUNK_DIM_X / TILE_DIM_X;
constexpr size_t STAGES  = TILES_Y * TILES_X;

constexpr size_t BUFFS_NUM      = 2;
constexpr size_t BUFF_DIM_Y     = TILE_DIM_Y;
constexpr size_t BUFF_DIM_X     = TILE_DIM_X;
constexpr size_t BUFF_SIZE      = BUFF_DIM_Y * BUFF_DIM_X;
constexpr size_t BUFF_SIZE_TOTAL = BUFF_SIZE * BUFFS_NUM;

// Input buffer (rowwise BF16/FP16/FP32)
constexpr size_t BUFF_IN_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_IN_DIM_X = BUFF_DIM_X;
constexpr size_t BUFF_IN_SIZE  = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;

#if MXFP4_SIMULATE_WITH_FP8
// Simulation path: one fp8e4m3 per logical matrix element.
constexpr size_t BUFF_OUT_DIM_Y  = BUFF_DIM_Y;
constexpr size_t BUFF_OUT_DIM_X  = BUFF_DIM_X;
constexpr size_t BUFF_OUT_SIZE   = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

constexpr size_t BUFF_OUT_T_DIM_Y = BUFF_DIM_X;
constexpr size_t BUFF_OUT_T_DIM_X = BUFF_DIM_Y;
constexpr size_t BUFF_OUT_T_SIZE  = BUFF_OUT_T_DIM_Y * BUFF_OUT_T_DIM_X;
#else
// Native FP4 path: packed fp4e2m1x2, 0.5 byte per logical element.
constexpr size_t BUFF_OUT_DIM_Y  = BUFF_DIM_Y;
constexpr size_t BUFF_OUT_DIM_X  = (BUFF_DIM_X * 4) / 8;
constexpr size_t BUFF_OUT_SIZE   = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

constexpr size_t BUFF_OUT_T_DIM_Y = BUFF_DIM_X;
constexpr size_t BUFF_OUT_T_DIM_X = (BUFF_DIM_Y * 4) / 8;
constexpr size_t BUFF_OUT_T_SIZE  = BUFF_OUT_T_DIM_Y * BUFF_OUT_T_DIM_X;
#endif

// Manual swizzling parameters to reduce SHMEM bank conflicts
constexpr size_t PACK_SIZE   = 8;
constexpr size_t WAVES       = SCALE_DIM / PACK_SIZE;

constexpr size_t SCALING_FACTORS_PER_TILE_X = TILE_DIM_X / SCALE_DIM;
constexpr size_t THREADS_X_ROWWISE          = SCALING_FACTORS_PER_TILE_X;
constexpr size_t THREADS_Y_ROWWISE          = THREADS_NUM / THREADS_X_ROWWISE;

constexpr size_t ITERATIONS_NORMAL    = BUFF_DIM_Y / THREADS_Y_ROWWISE;
constexpr size_t ITERATIONS_TRANSPOSE = BUFF_IN_DIM_Y / SCALE_DIM;
constexpr size_t BUFF_OUT_IT_OFFSET   = BUFF_OUT_T_DIM_X / ITERATIONS_TRANSPOSE;

static_assert(BUFF_DIM_Y >= SCALE_DIM,
              "Number of buffer rows must be >= block size\0");
static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE,
              "Buffer rows must be >= rowwise thread count\0");

// Number of 4-bit elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM;  // 8 = 256 / 32

// ------------------------
// Scaling helpers (MXFP4)
// ------------------------
// FP4(E2M1) magnitude index -> FP8(E4M3) byte (OCP E4M3, bias = 7)
// 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
__device__ __constant__ uint8_t FP4_TO_FP8E4M3_LUT[16] = {
    // positive grid
    0x00,  // 0.0
    0x30,  // 0.5
    0x38,  // 1.0
    0x3C,  // 1.5
    0x40,  // 2.0
    0x44,  // 3.0
    0x48,  // 4.0
    0x4C,  // 6.0
    // negative grid (sign bit set)
    0x80,  // -0.0
    0xB0,  // -0.5
    0xB8,  // -1.0
    0xBC,  // -1.5
    0xC0,  // -2.0
    0xC4,  // -3.0
    0xC8,  // -4.0
    0xCC,  // -6.0
};

__device__ __forceinline__ uint8_t fp4_index_to_fp8_e4m3(uint8_t fp4_idx) {
  return FP4_TO_FP8E4M3_LUT[fp4_idx & 0x0F];
}

__device__ __forceinline__ float exp2f_rcp_e8m0(uint8_t b) {
  // returns 2^(127 - b).  Range: b in [0..255] => exponent in [-128..127]
  return ldexpf(1.0f, 127 - (int)b);
}

__device__ __forceinline__ float exp2f_e8m0(uint8_t b) {
  int e = (int)b - 127;         // e in [-127..128]
  if (e > 127) return FLT_MAX;  // b==255 -> e==128 (overflow in float)
  return ldexpf(1.0f, e);       // handles subnormals for e=-127
}

__device__ __forceinline__
void pack_fp8e4m3x2_from_fp4_indices(fp8e4m3x2 &out,
                                     uint8_t idx0,
                                     uint8_t idx1) {
  uint16_t b0 = fp4_index_to_fp8_e4m3(idx0);
  uint16_t b1 = fp4_index_to_fp8_e4m3(idx1);
  uint16_t packed = static_cast<uint16_t>(b0) |
                    (static_cast<uint16_t>(b1) << 8);
  reinterpret_cast<uint16_t &>(out) = packed;
}

/// Grid-level FP4 encoder with optional stochastic rounding.
///
/// Input `z` is already scaled: z = x * block_scale_inverse.
/// We quantize |z| onto {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
///
/// USE_SR == false  -> deterministic nearest neighbor on the FP4 grid
/// USE_SR == true   -> unbiased SR between the two neighboring grid points
template <bool USE_SR>
__device__ __forceinline__
uint8_t encode_fp4_index_from_scaled(float z, uint32_t rbits) {
  float a = fabsf(z);
  const int sign_bit = (z < 0.0f) ? 1 : 0;

  int   lo_idx = 0, hi_idx = 0;
  float lo_val = 0.0f, hi_val = 0.0f;

  // Bracket a between FP4 grid points (non-uniform)
  if (a <= 0.0f) {
    lo_idx = hi_idx = 0;
    lo_val = hi_val = 0.0f;
  } else if (a < 0.5f) {
    lo_idx = 0; hi_idx = 1;
    lo_val = 0.0f; hi_val = 0.5f;
  } else if (a < 1.0f) {
    lo_idx = 1; hi_idx = 2;
    lo_val = 0.5f; hi_val = 1.0f;
  } else if (a < 1.5f) {
    lo_idx = 2; hi_idx = 3;
    lo_val = 1.0f; hi_val = 1.5f;
  } else if (a < 2.0f) {
    lo_idx = 3; hi_idx = 4;
    lo_val = 1.5f; hi_val = 2.0f;
  } else if (a < 3.0f) {
    lo_idx = 4; hi_idx = 5;
    lo_val = 2.0f; hi_val = 3.0f;
  } else if (a < 4.0f) {
    lo_idx = 5; hi_idx = 6;
    lo_val = 3.0f; hi_val = 4.0f;
  } else if (a < 6.0f) {
    lo_idx = 6; hi_idx = 7;
    lo_val = 4.0f; hi_val = 6.0f;
  } else {
    // Saturate to 6.0
    lo_idx = hi_idx = 7;
    lo_val = hi_val = 6.0f;
  }

  int mag_idx;
  if (lo_idx == hi_idx) {
    // Saturation / exactly on a grid point
    mag_idx = lo_idx;
  } else if constexpr (USE_SR) {
    // Stochastic rounding between lo and hi.
    // p = (a - lo)/(hi - lo)
    float p = (a - lo_val) / (hi_val - lo_val);  // p in [0,1]
    // Make a uniform in [0,1) from high 24 bits of rbits
    float u = (rbits >> 8) * (1.0f / 16777216.0f);  // 2^24
    mag_idx = (u < p) ? hi_idx : lo_idx;
  } else {
    // Deterministic nearest neighbor with Round-To-Nearest-Even (RNE)
    float mid = 0.5f * (lo_val + hi_val);
    if (a == mid) {
        // Break tie towards even index
        mag_idx = (lo_idx % 2 == 0) ? lo_idx : hi_idx;
    } else {
        mag_idx = (a > mid) ? hi_idx : lo_idx;
    }
  }

  mag_idx &= 0x7;
  uint8_t fp4_idx = static_cast<uint8_t>(mag_idx | (sign_bit << 3));
  return fp4_idx;
}


__device__ __forceinline__
mxfp4_scale_t compute_decoding_scaling_factor(float block_amax, float S_enc) {
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max; // 6

  // Match python: "zero" check is absolute on block_amax
  if (block_amax <= 1.0e-9f) {
    return static_cast<mxfp4_scale_t>(0);  // exponent = -127 -> biased = 0
  }

  const float effective = (block_amax * S_enc) / fp4_max;

  float exponent_f = roundf(log2f(effective));
  int exponent_i = (int)exponent_f;
  exponent_i = max(-127, min(128, exponent_i));
  return static_cast<mxfp4_scale_t>(exponent_i + 127);
}


__device__ __forceinline__ mxfp4_scale_t compute_encoding_scaling_factor(const float block_amax,
                                                                         const float S_enc) {
  using namespace detail;

  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f

  // INVERSE LOGIC:
  // Original: (block_amax * S_enc) / fp4_max  --> Maps block down to normalized range
  // New:      fp4_max / (block_amax * S_enc)  --> Maps block up (reciprocal)
  // Note: Since S_enc is (fp4_max / global_amax), this effectively computes (global_amax / block_amax).
  const float effective_amax = fp4_max / (block_amax * S_enc);

  // Avoid division by zero issues (if block_amax is near 0, ratio is infinite).
  // In reciprocal mode, a zero block means we need the Maximum scale.
  if (block_amax <= 1.0e-9f) {
    return static_cast<mxfp4_scale_t>(255); // Max E8M0
  }

  // Match torch.ceil(log2(.)) behavior
  const float exponent_f = roundf(log2f(effective_amax));
  int exponent_i = static_cast<int>(exponent_f);

  // Clamp to valid E8M0 exponent range [-127, 128]
  exponent_i = max(-127, min(128, exponent_i));

  // Apply E8M0 bias (+127)
  return static_cast<mxfp4_scale_t>(exponent_i + 127);
}
// Global encode scaling factor used in the ablation path.
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f

  if (global_amax <= 1.0e-9f) {
    return fp4_max;
  }

  // Want global_amax * S_enc == fp4_max
  // => S_enc = fp4_max / global_amax
  return fp4_max / global_amax;
}

// RNG helper
__device__ __forceinline__ uint32_t get_rbits(RNG &rng, uint4 &random_uint4, int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    curanddx::uniform_bits dist;
    random_uint4 = dist.generate4(rng);
  }
  const uint32_t *const rbits_arr = reinterpret_cast<uint32_t *>(&random_uint4);
  return rbits_arr[rnd_idx++];
}

// ------------------------
// FP4 conversion helpers
// ------------------------

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b16 v0_bf16; \n\t"
        ".reg.b16 v1_bf16; \n\t"
        ".reg.b16 v2_bf16; \n\t"
        ".reg.b16 v3_bf16; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %2; \n\t"
        "mul.f32x2 v23, v23, %2; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %3; \n\t"
        "}\n"
        : "=h"(out_4x)
        : "l"(in_4x),
          "l"(reinterpret_cast<const uint64_t &>(scale)),
          "r"(rbits));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_rn(
    const uint64_t in_4x, const float2 scale, const uint32_t /*rbits*/) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  uint32_t out_4x = 0;  // container for 16 bits
  if constexpr (is_blackwell) {
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b16 v0_bf16; \n\t"
        ".reg.b16 v1_bf16; \n\t"
        ".reg.b16 v2_bf16; \n\t"
        ".reg.b16 v3_bf16; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        ".reg.b8  f0; \n\t"
        ".reg.b8  f1; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %2; \n\t"
        "mul.f32x2 v23, v23, %2; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3; \n\t"
        "mov.b32 %0, {f0, f1, f0, f1}; \n\t"
        "}\n"
        : "=r"(out_4x)
        : "l"(in_4x),
          "l"(reinterpret_cast<const uint64_t &>(scale)));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(in_4x, scale, rbits);
  } else {
    return mul_cvt_bf16_to_fp4_4x_with_rn(in_4x, scale, rbits);
  }
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        "mov.b64 {v0, v1}, %1; \n\t"
        "mov.b64 {v2, v3}, %2; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %3; \n\t"
        "mul.f32x2 v23, v23, %3; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %4; \n\t"
        "}\n"
        : "=h"(out_4x)
        : "l"(reinterpret_cast<const uint64_t &>(in01)),
          "l"(reinterpret_cast<const uint64_t &>(in23)),
          "l"(reinterpret_cast<const uint64_t &>(scale)),
          "r"(rbits));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_rn(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t /*rbits*/) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  uint32_t out_4x = 0;
  if constexpr (is_blackwell) {
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        ".reg.b8  f0; \n\t"
        ".reg.b8  f1; \n\t"
        "mov.b64 {v0, v1}, %1; \n\t"
        "mov.b64 {v2, v3}, %2; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %3; \n\t"
        "mul.f32x2 v23, v23, %3; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3; \n\t"
        "mov.b32 %0, {f0, f1, f0, f1}; \n\t"
        "}\n"
        : "=r"(out_4x)
        : "l"(reinterpret_cast<const uint64_t &>(in01)),
          "l"(reinterpret_cast<const uint64_t &>(in23)),
          "l"(reinterpret_cast<const uint64_t &>(scale)));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(in01, in23, scale, rbits);
  } else {
    return mul_cvt_fp32_to_fp4_4x_with_rn(in01, in23, scale, rbits);
  }
}

// ---------------------------------
// 1D MXFP4 quantize + transpose
// ---------------------------------

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING,
          bool RETURN_TRANSPOSE, bool USE_GLOBAL_SCALE, bool ENCODE_CENTRIC>
__global__ void __launch_bounds__(THREADS_NUM)
mxfp4_transpose_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                       const __grid_constant__ CUtensorMap tensor_map_output,
                       const __grid_constant__ CUtensorMap tensor_map_output_t,
                       mxfp4_scale_t *const scales_ptr,
                       mxfp4_scale_t *const scales_t_ptr,
                       const float *noop,
                       const float *const amax_rowwise_ptr,
                       const float *const amax_colwise_ptr,
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

  const size_t rng_sequence =
      threadIdx.x + blockIdx.x * THREADS_NUM +
      blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed   = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG   rng(rng_seed, rng_sequence, rng_offset);
  curanddx::uniform_bits dist;
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? dist.generate4(rng)
                                               : uint4{0, 0, 0, 0};
  int rnd_idx = 0;

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;

  const size_t chunk_rows = rows - block_offset_Y;

  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;
  const size_t scales_block_offset_Y_t       = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_X_t       = blockIdx.y * SCALES_PER_CHUNK_Y;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_X_colwise = threadIdx.x;
  const size_t tid_Y_t       = tid_X_colwise;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const size_t row_base_colwise = block_offset_Y;
  const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_t       = scales_block_offset_Y_t + tid_Y_t;
  const size_t scales_offset_X_t       = scales_block_offset_X_t;

  const size_t SFs_per_row = cols / SCALE_DIM;

  const bool rowwise_scale_is_within_bounds_X = (scales_offset_X_rowwise < SFs_per_row);
  const bool colwise_scale_is_within_bounds_Y = (scales_offset_Y_t < cols);

  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  // Helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group  = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems       = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
#if MXFP4_SIMULATE_WITH_FP8
  // FP8: 1 byte per element
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total /* bytes */, TMA_SHMEM_ALIGNMENT);
#else
  // FP4: 0.5 byte per element
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
#endif

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_mem_rowwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_rowwise_scales = 0;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  constexpr float fp4_max = 6.0f;  // 6.0f

#if MXFP4_SIMULATE_WITH_FP8
  // each element is 1 byte (fp8e4m3)
  fp8e4m3 *out_data_sh =
      reinterpret_cast<fp8e4m3 *>(dshmem + in_mem);
  fp8e4m3 *out_t_data_sh =
      reinterpret_cast<fp8e4m3 *>(dshmem + in_mem + out_mem_rowwise_data);
#else
  fp4e2m1x2 *out_data_sh =
      reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh =
      reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);
#endif

  mxfp4_scale_t *out_rowwise_scales_sh = reinterpret_cast<mxfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  mxfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<mxfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);

  IType *cached_act_sh = in_sh;  // reuse input buffer as cache

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  // Global encode scaling factors (optional)
  // [NEW] If USE_GLOBAL_SCALE is off, we default to fp4_max (6.0) so that 
  // the resulting block scale is absolute (block_amax). This is because 
  // the GEMM kernel always applies a 1/36 factor to the alpha parameter.
  float S_enc_rowwise =  (USE_GLOBAL_SCALE)
                        ? compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr): fp4_max;

  float S_enc_colwise =  (USE_GLOBAL_SCALE)
                        ? compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr): fp4_max;

  float thread_amax = 0.0f;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input,
                    block_offset_X, block_offset_Y,
                    shmem_buff_size, &mbar[0], is_master_thread);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff       = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in   = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out  = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff           = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y     = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X     = block_offset_X;
      const size_t next_buff_offset    = next_buff * BUFF_IN_SIZE;

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
        const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;
        const size_t in_thread_offset_X = thread_offset_X_colwise;

        const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
        const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

        const size_t shmem_offset_base_colwise_in =
            buff_offset_in +
            in_thread_offset_Y * BUFF_IN_DIM_X +
            in_thread_offset_X;
        const size_t shmem_offset_base_colwise_out_t =
            buff_offset_out_t +
            out_t_thread_offset_Y * BUFF_OUT_T_DIM_X +
            out_t_thread_offset_X;

        block_amax = 0.0f;
        float   in_compute_colwise[SCALE_DIM];
        IType   in_colwise_IType[SCALE_DIM];

        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise =
                shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            in_colwise_IType[i] = in_sh[shmem_offset_colwise];
            block_amax_f16 = __hmax(block_amax_f16,
                                    __habs(in_colwise_IType[i]));
          }
          block_amax = static_cast<float>(block_amax_f16);
        } else {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise =
                shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
            if constexpr (COMPUTE_ACTIVATIONS) {
              elt = OP(elt, {});
            }
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            if constexpr (IS_CACHED_ACT_OP) {
              cached_act_sh[shmem_offset_colwise] =
                  static_cast<IType>(elt);
            }
            if constexpr (COMPUTE_ACTIVATIONS) {
              const bool row_out_of_bounds_colwise =
                  (row_base_colwise + stage_offset_Y + i >= rows);
              const bool out_of_bounds =
                  (col_out_of_bounds_colwise || row_out_of_bounds_colwise);
              if (!out_of_bounds) {
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
            } else {
              block_amax = fmaxf(block_amax, fabsf(elt));
            }
            in_compute_colwise[i] = elt;
          }
        }

        mxfp4_scale_t S_b_fp8;
        float block_scale_inverse; 

        if constexpr (ENCODE_CENTRIC) {
            // --- ENCODE CENTRIC PATH ---
            // 1. Compute Reciprocal Scale (mimicking structure)
            //    Returns approx: global_amax / block_amax
            mxfp4_scale_t mult_bits = compute_encoding_scaling_factor(block_amax, S_enc_colwise);
            
            // 2. Use Linear exponent for local math (Multiplier)
            block_scale_inverse = S_enc_colwise * exp2f_e8m0(mult_bits);

            // 3. FLIP exponent for storage (Divisor)
            // GEMM expects a Divisor. Since Mult * Div = 1, E_div = 254 - E_mult.
            int flipped = 254 - (int)mult_bits;
            S_b_fp8 = static_cast<mxfp4_scale_t>(max(0, min(255, flipped)));
        } 
        else {
            // --- DECODE CENTRIC PATH (Standard Nvidia) ---
            // 1. Compute Standard Scale
            //    Returns approx: block_amax / global_amax
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc_colwise);
            
            // 2. Apply Scale (Inverse required)
            //    Math: S_enc * (1 / stored_scale)
            //        = (6.0/global) * (global/block)
            //        = 6.0/block
            block_scale_inverse = S_enc_colwise * exp2f_rcp_e8m0(S_b_fp8);
        }

        const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_b_fp8;

        const float2 block_scale_inverse_2x { block_scale_inverse, block_scale_inverse };

#if MXFP4_SIMULATE_WITH_FP8
        fp8e4m3 *out_base =
            &out_t_data_sh[shmem_offset_base_colwise_out_t];

#pragma unroll
        for (int i = 0; i < SCALE_DIM; i += 2) {
          // Get scaled values z = x * block_scale_inverse
          float z0, z1;
          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            // BF16 / FP16 path: use in_colwise_IType[]
            z0 = static_cast<float>(in_colwise_IType[i + 0]) * block_scale_inverse;
            z1 = static_cast<float>(in_colwise_IType[i + 1]) * block_scale_inverse;
          } else {
            // FP32 (or cached activation) path: in_compute_colwise[] holds x
            z0 = in_compute_colwise[i + 0] * block_scale_inverse;
            z1 = in_compute_colwise[i + 1] * block_scale_inverse;
          }

          // Per-value random bits (SR or ignored if not enabled)
          const uint32_t rbits0 = get_rbits(rng, random_uint4, rnd_idx);
          const uint32_t rbits1 = get_rbits(rng, random_uint4, rnd_idx);

          // Quantize to FP4 grid (E2M1) and map to FP8(E4M3)
          uint8_t fp4_idx0 =
              encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(z0, rbits0);
          uint8_t fp4_idx1 =
              encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(z1, rbits1);

          fp8e4m3x2 out_pair;
          pack_fp8e4m3x2_from_fp4_indices(out_pair, fp4_idx0, fp4_idx1);

          reinterpret_cast<fp8e4m3x2 &>(out_base[i]) = out_pair;
        }
#else
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
            const float2 in01 =
                *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
            const float2 in23 =
                *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
            regs[e] = mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                in01, in23, block_scale_inverse_2x, rbits);
          }
        }

        const int group = thread_lane / 16;
        uint32_t val[4];
        uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

        switch (group) {
          case 0:
            val[0] = regs_4x[0];
            val[1] = regs_4x[1];
            val[2] = regs_4x[2];
            val[3] = regs_4x[3];
            break;
          case 1:
            val[0] = regs_4x[1];
            val[1] = regs_4x[0];
            val[2] = regs_4x[3];
            val[3] = regs_4x[2];
            break;
        }

        uint32_t *out_t_data_sh_as_uint32_t =
            reinterpret_cast<uint32_t *>(
                &out_t_data_sh[shmem_offset_base_colwise_out_t]);

        out_t_data_sh_as_uint32_t[group]         = val[0];
        out_t_data_sh_as_uint32_t[(group ^ 1)]   = val[1];
        out_t_data_sh_as_uint32_t[group + 2]     = val[2];
        out_t_data_sh_as_uint32_t[(group ^ 1)+2] = val[3];
#endif  // MXFP4_SIMULATE_WITH_FP8
      }
    }  // RETURN_TRANSPOSE

    // ROWWISE scaling
    {
      const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t it_thread_offset_Y_rowwise =
            thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const size_t shmem_offset_base_rowwise_in =
            buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
        const size_t shmem_offset_base_rowwise_out =
            buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

        const size_t it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;
        float in_compute_rowwise[SCALE_DIM];
        Vec<IType, PACK_SIZE> in_cached[WAVES];
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x =
              {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx =
                ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx =
                thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise =
                shmem_offset_base_rowwise_in + swizzled_thread_idx;
            in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x,
                              in_IType[w].data.elt[e]);
            }
          }
          block_amax =
              static_cast<float>(__hmax(__habs(thread_amax_2x.x),
                                        __habs(thread_amax_2x.y)));
        } else if constexpr (IS_CACHED_ACT_OP) {
          __syncthreads();
          IType2 thread_amax_2x =
              {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx =
                ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx =
                thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise =
                shmem_offset_base_rowwise_in + swizzled_thread_idx;

            const bool row_out_of_bounds_rowwise =
                (row_base_rowwise + it_offset_Y >= rows);
            const bool swizzled_col_out_of_bounds =
                (block_offset_X + swizzled_thread_idx >= cols);
            const bool out_of_bounds =
                (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);

            in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
            if (!out_of_bounds) {
              if constexpr (std::is_same_v<IType, float>) {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; ++e) {
                  block_amax = fmaxf(block_amax,
                                     fabsf(in_cached[w].data.elt[e]));
                }
              } else {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; e += 2) {
                  const IType2 in_cached_2x =
                      {in_cached[w].data.elt[e],
                       in_cached[w].data.elt[e + 1]};
                  ptx::abs_max_2x(thread_amax_2x, thread_amax_2x,
                                  in_cached_2x);
                }
              }
            }
          }
          if constexpr (!std::is_same_v<IType, float>) {
            block_amax =
                static_cast<float>(__hmax(__habs(thread_amax_2x.x),
                                          __habs(thread_amax_2x.y)));
          }
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx =
                ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx =
                thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise =
                shmem_offset_base_rowwise_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            in.load_from(&in_sh[shmem_offset_rowwise]);

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
                const bool row_out_of_bounds_rowwise =
                    (row_base_rowwise + it_offset_Y >= rows);
                const bool swizzled_col_out_of_bounds =
                    (block_offset_X + swizzled_thread_idx >= cols);
                const bool out_of_bounds =
                    (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
                if (!out_of_bounds) {
                  block_amax = fmaxf(block_amax, fabsf(elt));
                }
              } else {
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
              in_compute_rowwise[j] = elt;
            }
          }
        }
        #if MXFP4_DEBUG_PRINTS
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid_X_rowwise == 0 && tid_Y_rowwise == 0 && stage == 0 && it == 0) {
             printf("\n[KERNEL BLOCK 0] BlockMax: %.6f\n", block_amax);
             printf("[KERNEL BLOCK 0] S_enc: %.6f\n", S_enc_rowwise);
             
             // Replicate the math to see intermediate values
             constexpr float fp4_max = 6.0f;
             float effective = (block_amax * S_enc_rowwise) / fp4_max;
             float log_val = log2f(effective);
             float ceil_val = ceilf(log_val);
             
             printf("[KERNEL BLOCK 0] Effective: %.6f\n", effective);
             printf("[KERNEL BLOCK 0] Log2(Eff): %.6f\n", log_val);
             printf("[KERNEL BLOCK 0] Ceil: %.6f\n", ceil_val);
        }
        #endif
        mxfp4_scale_t S_b_fp8;
        float block_scale_inverse;

        if constexpr (ENCODE_CENTRIC) {
            // [Encode-Centric]
            // Calculate Multiplier directly: S ~ (FP4_MAX / (block_amax * S_enc))
            mxfp4_scale_t mult_bits = compute_encoding_scaling_factor(block_amax, S_enc_rowwise);
            
            // 2. Use Linear exponent for local math (Multiplier)
            block_scale_inverse = S_enc_rowwise * exp2f_e8m0(mult_bits);

            // 3. FLIP exponent for storage (Divisor)
            // GEMM expects a Divisor. Since Mult * Div = 1, E_div = 254 - E_mult.
            int flipped = 254 - (int)mult_bits;
            S_b_fp8 = static_cast<mxfp4_scale_t>(max(0, min(255, flipped)));
        } else {
            // [Decode-Centric / Nvidia]
            // Calculate Divisor: S ~ (block_amax * S_enc) / FP4_MAX
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc_rowwise);
            
            // Apply Inverse (Reciprocal)
            // effective_scale = S_enc * (1 / stored_scale)
            block_scale_inverse = S_enc_rowwise * exp2f_rcp_e8m0(S_b_fp8);
        }

        // 2. Global Memory Store (Shared logic)
        const size_t scales_offset_Y =
            scales_offset_Y_rowwise + stage * BUFF_DIM_Y +
            it * THREADS_Y_ROWWISE;
        const size_t scales_offset_X = scales_offset_X_rowwise;
        const size_t scale_idx_global =
            scales_offset_Y * scale_stride + scales_offset_X;

        const bool rowwise_scale_is_within_bounds_Y =
            (stage_rowwise_scales_offset_Y +
             it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;

        if (rowwise_scale_is_within_bounds_X &&
            rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_b_fp8;
        }

        // 3. Prepare factor for quantization loop
        const float2 block_scale_inverse_2x { block_scale_inverse,
                                              block_scale_inverse };

#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
#if MXFP4_SIMULATE_WITH_FP8
          const size_t swizzled_group_idx =
              ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx =
              swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise =
              shmem_offset_base_rowwise_out + swizzled_idx;

          fp8e4m3 *out_row = &out_data_sh[shmem_offset_rowwise];

#pragma unroll
          for (int e = 0; e < PACK_SIZE; e += 2) {
            const int j = w * PACK_SIZE + e;

            // 1. Get the scaled values z = x * block_scale_inverse.
            float v0, v1;

            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              // BF16 / FP16: load from the same swizzled region we used for amax.
              const size_t swizzled_idx =
                  swizzled_group_idx + thread_offset_X_rowwise;
              const size_t shmem_offset_rowwise_in =
                  shmem_offset_base_rowwise_in + swizzled_idx;

              const size_t in_sh_offset = shmem_offset_rowwise_in + e;
              v0 = static_cast<float>(in_sh[in_sh_offset + 0]) * block_scale_inverse;
              v1 = static_cast<float>(in_sh[in_sh_offset + 1]) * block_scale_inverse;
            } else {
              v0 = in_compute_rowwise[j + 0] * block_scale_inverse;
              v1 = in_compute_rowwise[j + 1] * block_scale_inverse;
            }

            // 2. Quantize to FP4 grid (E2M1) in float.
            const uint32_t rbits0 = get_rbits(rng, random_uint4, rnd_idx);
            const uint32_t rbits1 = get_rbits(rng, random_uint4, rnd_idx);

            // Quantize to FP4 grid (E2M1) and map to FP8(E4M3)
            uint8_t fp4_idx0 =
                encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(v0, rbits0);
            uint8_t fp4_idx1 =
                encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(v1, rbits1);

            // 3. Map FP4 indices to FP8 E4M3 codes.
            fp8e4m3x2 out_pair;
            pack_fp8e4m3x2_from_fp4_indices(out_pair, fp4_idx0, fp4_idx1);

            reinterpret_cast<fp8e4m3x2 &>(out_row[e]) = out_pair;
          }
#else
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
                  make_float2(in_compute_rowwise[j],
                              in_compute_rowwise[j + 1]);
              const float2 in23 =
                  make_float2(in_compute_rowwise[j + 2],
                              in_compute_rowwise[j + 3]);
              out.data.elt[e] =
                  mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                      in01, in23, block_scale_inverse_2x, rbits);
            }
          }
          const size_t swizzled_group_idx =
              ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx =
              swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise =
              shmem_offset_base_rowwise_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset_rowwise]);
#endif  // MXFP4_SIMULATE_WITH_FP8
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

      const size_t global_offset_Y_t = block_offset_Y_t;
      const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output),
          global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_offset_out]));

      if constexpr (RETURN_TRANSPOSE) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t),
            global_offset_X_t, global_offset_Y_t,
            reinterpret_cast<uint64_t *>(&out_t_data_sh[buff_offset_out_t]));
      }

      ptx::cp_async_bulk_commit_group();
    }
  }  // stage loop

  // Vectorized colwise scale store
  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
    using ScalesVec = Vec<mxfp4_scale_t, SCALES_PER_CHUNK_Y>;
    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
    ScalesVec &scales_vec =
        *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);
    const size_t scale_idx_global =
        scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
    const size_t count =
        (chunk_rows >= CHUNK_DIM_Y)
            ? SCALES_PER_CHUNK_Y
            : (chunk_rows / SCALE_DIM);
    mxfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
    constexpr size_t vec_bytes =
        SCALES_PER_CHUNK_Y * sizeof(mxfp4_scale_t);
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

// ---------------------------------
// 2D MXFP4 quantize + transpose
// ---------------------------------

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING,
          bool RETURN_TRANSPOSE, bool USE_GLOBAL_SCALE, bool ENCODE_CENTRIC>
__global__ void __launch_bounds__(THREADS_NUM)
mxfp4_transpose_kernel_2D(const __grid_constant__ CUtensorMap tensor_map_input,
                          const __grid_constant__ CUtensorMap tensor_map_output,
                          const __grid_constant__ CUtensorMap tensor_map_output_t,
                          mxfp4_scale_t *const scales_ptr,
                          mxfp4_scale_t *const scales_t_ptr,
                          const float *noop,
                          const float *const amax_rowwise_ptr,
                          const float *const amax_colwise_ptr,
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

  const size_t rng_sequence =
      threadIdx.x + blockIdx.x * THREADS_NUM +
      blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed   = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG   rng(rng_seed, rng_sequence, rng_offset);
  curanddx::uniform_bits dist;
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? dist.generate4(rng)
                                               : uint4{0, 0, 0, 0};
  int rnd_idx = 0;
  constexpr float fp4_max = 6.0f;  // 6.0f
  // 2D block-based scaling
  constexpr size_t BLOCK_DIM          = 32;
  constexpr size_t BLOCKS_PER_TILE_Y  = TILE_DIM_Y / BLOCK_DIM;   // 1
  constexpr size_t BLOCKS_PER_TILE_X  = TILE_DIM_X / BLOCK_DIM;   // 4
  constexpr size_t ITERATIONS_BLOCK   = 1;
  constexpr size_t BLOCKS_PER_WARP    =
      BLOCKS_PER_TILE_X / (THREADS_NUM / 32);  // 4 / (128/32) = 1

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;

  const size_t chunk_rows = rows - block_offset_Y;

  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;
  const size_t scales_block_offset_Y_t       = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_X_t       = blockIdx.y * SCALES_PER_CHUNK_Y;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_X_colwise = threadIdx.x;
  const size_t tid_Y_t       = tid_X_colwise;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t scales_offset_Y_rowwise =
      scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise =
      scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_t =
      scales_block_offset_Y_t + tid_Y_t;
  const size_t scales_offset_X_t =
      scales_block_offset_X_t;

  const size_t SFs_per_row = cols / SCALE_DIM;

  const bool rowwise_scale_is_within_bounds_X =
      (scales_offset_X_rowwise < SFs_per_row);
  const bool colwise_scale_is_within_bounds_Y =
      (scales_offset_Y_t < cols);

  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group  = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems       = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
#if MXFP4_SIMULATE_WITH_FP8
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total /* bytes */, TMA_SHMEM_ALIGNMENT);
#else
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
#endif

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_mem_rowwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_rowwise_scales = 0;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  IType *in_sh = reinterpret_cast<IType *>(dshmem);
#if MXFP4_SIMULATE_WITH_FP8
  fp8e4m3 *out_data_sh =
      reinterpret_cast<fp8e4m3 *>(dshmem + in_mem);
  fp8e4m3 *out_t_data_sh =
      reinterpret_cast<fp8e4m3 *>(dshmem + in_mem + out_mem_rowwise_data);
#else
  fp4e2m1x2 *out_data_sh =
      reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh =
      reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);
#endif

  mxfp4_scale_t *out_rowwise_scales_sh = reinterpret_cast<mxfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  mxfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<mxfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);

  IType *cached_act_sh = in_sh;

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  // Global scaling toggles
  float S_enc_rowwise = ( USE_GLOBAL_SCALE)
                        ? compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr): fp4_max;

  float S_enc_colwise = ( USE_GLOBAL_SCALE)
                        ? compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr): fp4_max;

  const size_t warp_id       = threadIdx.x / 32;
  const size_t lane_id       = threadIdx.x % 32;
  float        thread_amax   = 0.0f;
  const size_t block_in_warp = lane_id / BLOCKS_PER_WARP;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  __shared__ __align__(16) float block_amax_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1];

  auto warp_reduce_amax = [](float value) {
#pragma unroll
    for (int delta = 16; delta >= 1; delta >>= 1) {
      float other = __shfl_xor_sync(0xffffffff, value, delta);
      value = fmaxf(value, other);
    }
    return value;
  };

  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input,
                    block_offset_X, block_offset_Y,
                    shmem_buff_size, &mbar[0], is_master_thread);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff       = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in   = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out  = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    if (next_stage < STAGES) {
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff           = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y     = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X     = block_offset_X;
      const size_t next_buff_offset    = next_buff * BUFF_IN_SIZE;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input,
                        global_offset_X, global_offset_Y,
                        shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;

    // First pass: compute 2D block amaxes
#pragma unroll
    for (size_t block_iter = 0; block_iter < ITERATIONS_BLOCK; ++block_iter) {
      IType2 thread_amax_2x =
          {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
      const size_t block_in_tile_y = block_iter;
      const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;

      if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
        for (int elem = 0; elem < BLOCK_DIM; elem += 2) {
          const size_t elem_0_row = block_iter * BLOCK_DIM + elem;
          const size_t elem_1_row = elem_0_row + 1;
          const size_t elem_0_col =
              warp_id * BLOCKS_PER_WARP * BLOCK_DIM + lane_id;
          const size_t elem_1_col = elem_0_col;

          const size_t shmem_offset_0 =
              buff_offset_in + elem_0_row * BUFF_IN_DIM_X + elem_0_col;
          const size_t shmem_offset_1 =
              buff_offset_in + elem_1_row * BUFF_IN_DIM_X + elem_1_col;

          IType2 val_2x;
          val_2x.x = in_sh[shmem_offset_0];
          val_2x.y = in_sh[shmem_offset_1];
          ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, val_2x);
        }
        thread_amax =
            static_cast<float>(__hmax(__habs(thread_amax_2x.x),
                                      __habs(thread_amax_2x.y)));
      } else {
        for (int elem = 0; elem < BLOCK_DIM; ++elem) {
          const size_t elem_row = block_iter * BLOCK_DIM + elem;
          const size_t elem_col =
              warp_id * BLOCKS_PER_WARP * BLOCK_DIM + lane_id;

          const bool row_out_of_bounds =
              (block_offset_Y + stage_offset_Y + elem_row >= rows);
          const bool col_out_of_bounds =
              (block_offset_X + elem_col >= cols);
          if (!row_out_of_bounds && !col_out_of_bounds) {
            const size_t shmem_offset =
                buff_offset_in + elem_row * BUFF_IN_DIM_X + elem_col;
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
            thread_amax = fmaxf(thread_amax, fabsf(elt));
          }
        }
      }

      block_amax = warp_reduce_amax(thread_amax);

      if (lane_id == 0) {
        block_amax_matrix[block_in_tile_y][block_in_tile_x] = block_amax;
      }
    }

    __syncthreads();

    // COLWISE scaling (transpose)
    if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
        const size_t block_in_tile_y = it;
        const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;

        const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;
        const size_t in_thread_offset_X = thread_offset_X_colwise;

        const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
        const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

        const size_t shmem_offset_base_colwise_in =
            buff_offset_in +
            in_thread_offset_Y * BUFF_IN_DIM_X +
            in_thread_offset_X;
        const size_t shmem_offset_base_colwise_out_t =
            buff_offset_out_t +
            out_t_thread_offset_Y * BUFF_OUT_T_DIM_X +
            out_t_thread_offset_X;

        block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];
        float in_compute_colwise[SCALE_DIM];
        IType in_colwise_IType[SCALE_DIM];

        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise =
                shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            in_colwise_IType[i] = in_sh[shmem_offset_colwise];
          }
        } else {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise =
                shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
            if constexpr (COMPUTE_ACTIVATIONS) {
              elt = OP(elt, {});
            }
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            if constexpr (IS_CACHED_ACT_OP) {
              cached_act_sh[shmem_offset_colwise] =
                  static_cast<IType>(elt);
            }
            in_compute_colwise[i] = elt;
          }
        }

        mxfp4_scale_t S_b_fp8;
        float block_scale_inverse;

        if constexpr (ENCODE_CENTRIC) {
            // [Encode-Centric]
            // Calculate Multiplier: S ~ (FP4_MAX / (block_amax * S_enc))
            mxfp4_scale_t mult_bits = compute_encoding_scaling_factor(block_amax, S_enc_colwise);
            
            // 2. Use Linear exponent for local math (Multiplier)
            block_scale_inverse = S_enc_colwise * exp2f_e8m0(mult_bits);

            // 3. FLIP exponent for storage (Divisor)
            // GEMM expects a Divisor. Since Mult * Div = 1, E_div = 254 - E_mult.
            int flipped = 254 - (int)mult_bits;
            S_b_fp8 = static_cast<mxfp4_scale_t>(max(0, min(255, flipped)));
        } else {
            // [Decode-Centric / Nvidia]
            // Calculate Divisor: S ~ (block_amax * S_enc) / FP4_MAX
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc_colwise);

            // Apply Inverse (Reciprocal)
            // effective_scale = S_enc * (1 / stored_scale)
            block_scale_inverse = S_enc_colwise * exp2f_rcp_e8m0(S_b_fp8);
        }

        // 2. Store to Shared Memory (Shared logic)
        const size_t scale_idx_sh =
            tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_b_fp8;

        // 3. Prepare factor for quantization loop
        const float2 block_scale_inverse_2x { block_scale_inverse,
                                              block_scale_inverse };

#if MXFP4_SIMULATE_WITH_FP8
        fp8e4m3 *out_base =
            &out_t_data_sh[shmem_offset_base_colwise_out_t];

#pragma unroll
        for (int i = 0; i < SCALE_DIM; i += 2) {
          float z0, z1;
          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            // BF16 / FP16 path: in_colwise_IType[] filled above
            z0 = static_cast<float>(in_colwise_IType[i + 0]) * block_scale_inverse;
            z1 = static_cast<float>(in_colwise_IType[i + 1]) * block_scale_inverse;
          } else {
            // FP32 / cached activation path
            z0 = in_compute_colwise[i + 0] * block_scale_inverse;
            z1 = in_compute_colwise[i + 1] * block_scale_inverse;
          }

          const uint32_t rbits0 = get_rbits(rng, random_uint4, rnd_idx);
          const uint32_t rbits1 = get_rbits(rng, random_uint4, rnd_idx);

          uint8_t fp4_idx0 =
              encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(z0, rbits0);
          uint8_t fp4_idx1 =
              encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(z1, rbits1);

          fp8e4m3x2 out_pair;
          pack_fp8e4m3x2_from_fp4_indices(out_pair, fp4_idx0, fp4_idx1);

          reinterpret_cast<fp8e4m3x2 &>(out_base[i]) = out_pair;
        }
#else
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
            const float2 in01 =
                *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
            const float2 in23 =
                *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
            regs[e] = mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                in01, in23, block_scale_inverse_2x, rbits);
          }
        }

        const int group = thread_lane / 16;
        uint32_t val[4];
        uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

        switch (group) {
          case 0:
            val[0] = regs_4x[0];
            val[1] = regs_4x[1];
            val[2] = regs_4x[2];
            val[3] = regs_4x[3];
            break;
          case 1:
            val[0] = regs_4x[1];
            val[1] = regs_4x[0];
            val[2] = regs_4x[3];
            val[3] = regs_4x[2];
            break;
        }

        uint32_t *out_t_data_sh_as_uint32_t =
            reinterpret_cast<uint32_t *>(
                &out_t_data_sh[shmem_offset_base_colwise_out_t]);

        out_t_data_sh_as_uint32_t[group]         = val[0];
        out_t_data_sh_as_uint32_t[(group ^ 1)]   = val[1];
        out_t_data_sh_as_uint32_t[group + 2]     = val[2];
        out_t_data_sh_as_uint32_t[(group ^ 1)+2] = val[3];
#endif  // MXFP4_SIMULATE_WITH_FP8
      }
    }

    // ROWWISE scaling
    {
      const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t block_in_tile_y = it;
        const size_t block_in_tile_x = tid_X_rowwise;

        const size_t it_thread_offset_Y_rowwise =
            thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const size_t shmem_offset_base_rowwise_in =
            buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
        const size_t shmem_offset_base_rowwise_out =
            buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

        block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];
        float in_compute_rowwise[SCALE_DIM];
        Vec<IType, PACK_SIZE> in_cached[WAVES];
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx =
                ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx =
                thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise =
                shmem_offset_base_rowwise_in + swizzled_thread_idx;
            in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
          }
        } else if constexpr (IS_CACHED_ACT_OP) {
          __syncthreads();
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx =
                ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx =
                thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise =
                shmem_offset_base_rowwise_in + swizzled_thread_idx;
            in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
          }
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx =
                ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx =
                thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise =
                shmem_offset_base_rowwise_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            in.load_from(&in_sh[shmem_offset_rowwise]);

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
              in_compute_rowwise[j] = elt;
            }
          }
        }

        mxfp4_scale_t S_b_fp8;
        float block_scale_inverse;

        if constexpr (ENCODE_CENTRIC) {
            // [Encode-Centric]
            // Calculate Multiplier: S ~ (FP4_MAX / (block_amax * S_enc))
            mxfp4_scale_t mult_bits = compute_encoding_scaling_factor(block_amax, S_enc_rowwise);
            
            // 2. Use Linear exponent for local math (Multiplier)
            block_scale_inverse = S_enc_rowwise * exp2f_e8m0(mult_bits);

            // 3. FLIP exponent for storage (Divisor)
            // GEMM expects a Divisor. Since Mult * Div = 1, E_div = 254 - E_mult.
            int flipped = 254 - (int)mult_bits;
            S_b_fp8 = static_cast<mxfp4_scale_t>(max(0, min(255, flipped)));
        } else {
            // [Decode-Centric / Nvidia]
            // Calculate Divisor: S ~ (block_amax * S_enc) / FP4_MAX
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

            // Apply Inverse (Reciprocal)
            block_scale_inverse = S_enc_rowwise * exp2f_rcp_e8m0(S_b_fp8);
        }

        // 2. Global Memory Offsets (Shared logic)
        const size_t scales_offset_Y =
            scales_offset_Y_rowwise + stage * BUFF_DIM_Y +
            it * THREADS_Y_ROWWISE;
        const size_t scales_offset_X = scales_offset_X_rowwise;
        const size_t scale_idx_global =
            scales_offset_Y * scale_stride + scales_offset_X;

        // 3. Bounds Check and Store (Shared logic)
        const bool rowwise_scale_is_within_bounds_Y =
            (stage_rowwise_scales_offset_Y +
             it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;
        if (rowwise_scale_is_within_bounds_X &&
            rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_b_fp8;
        }

        // 4. Prepare factor for quantization loop
        const float2 block_scale_inverse_2x { block_scale_inverse,
                                              block_scale_inverse };

#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
#if MXFP4_SIMULATE_WITH_FP8
          const size_t swizzled_group_idx =
              ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx =
              swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise =
              shmem_offset_base_rowwise_out + swizzled_idx;

          fp8e4m3 *out_row = &out_data_sh[shmem_offset_rowwise];

#pragma unroll
        for (int e = 0; e < PACK_SIZE; e += 2) {
          const int j = w * PACK_SIZE + e;

          // 1. Get the scaled values z = x * block_scale_inverse.
          float v0, v1;

          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            const size_t swizzled_idx =
                swizzled_group_idx + thread_offset_X_rowwise;
            const size_t shmem_offset_rowwise_in =
                shmem_offset_base_rowwise_in + swizzled_idx;

            const size_t in_sh_offset = shmem_offset_rowwise_in + e;
            v0 = static_cast<float>(in_sh[in_sh_offset + 0]) * block_scale_inverse;
            v1 = static_cast<float>(in_sh[in_sh_offset + 1]) * block_scale_inverse;
          } else {
            v0 = in_compute_rowwise[j + 0] * block_scale_inverse;
            v1 = in_compute_rowwise[j + 1] * block_scale_inverse;
          }

          // 2. Quantize to FP4 grid (E2M1) in float.
          const uint32_t rbits0 = get_rbits(rng, random_uint4, rnd_idx);
          const uint32_t rbits1 = get_rbits(rng, random_uint4, rnd_idx);

          uint8_t fp4_idx0 =
              encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(v0, rbits0);
          uint8_t fp4_idx1 =
              encode_fp4_index_from_scaled<USE_STOCHASTIC_ROUNDING>(v1, rbits1);

          // 3. Map FP4 indices to FP8 E4M3 codes.
          fp8e4m3x2 out_pair;
          pack_fp8e4m3x2_from_fp4_indices(out_pair, fp4_idx0, fp4_idx1);

          reinterpret_cast<fp8e4m3x2 &>(out_row[e]) = out_pair;
        }
#else
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
                  make_float2(in_compute_rowwise[j],
                              in_compute_rowwise[j + 1]);
              const float2 in23 =
                  make_float2(in_compute_rowwise[j + 2],
                              in_compute_rowwise[j + 3]);
              out.data.elt[e] =
                  mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                      in01, in23, block_scale_inverse_2x, rbits);
            }
          }
          const size_t swizzled_group_idx =
              ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx =
              swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise =
              shmem_offset_base_rowwise_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset_rowwise]);
#endif  // MXFP4_SIMULATE_WITH_FP8
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

      const size_t global_offset_Y_t = block_offset_Y_t;
      const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output),
          global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_offset_out]));

      if constexpr (RETURN_TRANSPOSE) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t),
            global_offset_X_t, global_offset_Y_t,
            reinterpret_cast<uint64_t *>(&out_t_data_sh[buff_offset_out_t]));
      }

      ptx::cp_async_bulk_commit_group();
    }
  }  // stage loop

  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
    using ScalesVec = Vec<mxfp4_scale_t, SCALES_PER_CHUNK_Y>;
    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
    ScalesVec &scales_vec =
        *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);
    const size_t scale_idx_global =
        scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
    const size_t count =
        (chunk_rows >= CHUNK_DIM_Y)
            ? SCALES_PER_CHUNK_Y
            : (chunk_rows / SCALE_DIM);
    mxfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
    constexpr size_t vec_bytes =
        SCALES_PER_CHUNK_Y * sizeof(mxfp4_scale_t);
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

}  // namespace mxfp4_transpose

// ---------------------------------
// Host wrapper
// ---------------------------------

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          bool use_2d_quantization>
void mxfp4_quantize_transpose(const Tensor &input,
                              const Tensor *noop,
                              Tensor *output,
                              const QuantizationConfig *quant_config,
                              cudaStream_t stream) {
  using namespace mxfp4_transpose;
  using namespace ptx;

// #if MXFP4_SIMULATE_WITH_FP8
//   printf("mxfp4_quantize_transpose: SIM path (MXFP4_SIMULATE_WITH_FP8=1)\n");
// #else
//   printf("mxfp4_quantize_transpose: NATIVE path (MXFP4_SIMULATE_WITH_FP8=0)\n");
// #endif

//   printf("output->data.shape = {%zu, %zu}, dtype=%d\n",
//        output->data.shape[0], output->data.shape[1], int(output->data.dtype));

  // Config
  const bool use_stochastic_rounding =
      quant_config ? quant_config->stochastic_rounding : false;
  const bool use_global_scale =
      quant_config ? quant_config->global_scaling : false;
  const bool return_transpose = output->has_columnwise_data();
  const bool use_encode_centric = quant_config ? quant_config->encode_centric : false;
  // Validation
  checkCuDriverContext(stream);
  if (noop) {
    CheckNoopTensor(*noop, "cast_noop");
  }
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "MXFP4 output tensor must be allocated.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr,
             "Scaling tensor must be allocated.");

  if (return_transpose) {
    NVTE_CHECK(output->has_columnwise_data(),
               "MXFP4 transposed output tensor must be allocated.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Transposed scaling tensor must be allocated.");
  }

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  NVTE_CHECK(rows % 32 == 0,
             "Number of tensor rows must be a multiple of 32");
  NVTE_CHECK(cols % 32 == 0,
             "Number of tensor cols must be a multiple of 32");

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_NUM;

  const size_t scale_stride =
      static_cast<size_t>(output->scale_inv.shape[1]);
  const size_t scale_stride_transpose =
      return_transpose
          ? static_cast<size_t>(output->columnwise_scale_inv.shape[1])
          : 0;

  using mxfp4_scale_t = e8m0_t;
  mxfp4_scale_t *const scales_ptr =
      reinterpret_cast<mxfp4_scale_t *>(output->scale_inv.dptr);
  mxfp4_scale_t *const scales_transpose_ptr =
      return_transpose
          ? reinterpret_cast<mxfp4_scale_t *>(output->columnwise_scale_inv.dptr)
          : nullptr;

  const float *noop_ptr =
      (noop && noop->data.dptr)
          ? reinterpret_cast<const float *>(noop->data.dptr)
          : nullptr;

  const float *const amax_rowwise_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const float *const amax_colwise_ptr = reinterpret_cast<const float *>(output->columnwise_amax.dptr);
          
  

  // fprintf(stderr, "DEBUG: RowPtr=%p, ColPtr=%p\n", (void*)amax_rowwise_ptr, (void*)amax_colwise_ptr);
#if MXFP4_DEBUG_PRINTS
  if (amax_rowwise_ptr) {
      float r_val;
      // Must copy from GPU to CPU to print
      cudaMemcpy(&r_val, amax_rowwise_ptr, sizeof(float), cudaMemcpyDeviceToHost);
      fprintf(stderr, "DEBUG: RowVal=%f\n", r_val);
  } else {
      fprintf(stderr, "DEBUG: RowVal=[NULL]\n");
  }

  if (amax_colwise_ptr) {
      float c_val;
      cudaMemcpy(&c_val, amax_colwise_ptr, sizeof(float), cudaMemcpyDeviceToHost);
      fprintf(stderr, "DEBUG: ColVal=%f\n", c_val);
  } else {
      fprintf(stderr, "DEBUG: ColVal=[NULL]\n");
  }
#endif
  // fflush(stderr); // Force output immediately

  // RNG state
  const NVTETensor rng_state_tensor =
      (quant_config != nullptr) ? quant_config->rng_state : nullptr;
  const size_t *rng_state = nullptr;
  if (rng_state_tensor != nullptr) {
    Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
    rng_state = reinterpret_cast<const size_t *>(rng_state_te_tensor.data.dptr);
  }
  #if MXFP4_DEBUG_PRINTS
  fprintf(stderr,
    "[HOST mxfp4_quantize_transpose] quant_config=%p "
    "use_global_scale=%d use_encode_centric=%d use_sr=%d return_transpose=%d use_2d=%d\n",
    (void*)quant_config,
    (int)use_global_scale,
    (int)use_encode_centric,
    (int)use_stochastic_rounding,
    (int)return_transpose,
    (int)use_2d_quantization
  );

  if (quant_config) {
    fprintf(stderr,
      "[HOST quant_config fields] global_scaling=%d encode_centric=%d stochastic_rounding=%d rng_state=%p\n",
      (int)quant_config->global_scaling,
      (int)quant_config->encode_centric,
      (int)quant_config->stochastic_rounding,
      (void*)quant_config->rng_state
    );

    // ABI/layout sanity: dump raw bytes of QuantizationConfig (catches “old .so” issues)
    const unsigned char* p = reinterpret_cast<const unsigned char*>(quant_config);
    fprintf(stderr, "[HOST QuantizationConfig bytes] size=%zu:", sizeof(*quant_config));
    for (size_t i = 0; i < sizeof(*quant_config); ++i) fprintf(stderr, " %02x", p[i]);
    fprintf(stderr, "\n");
  }

  fprintf(stderr,
    "[HOST ptrs] amax_rowwise_ptr=%p amax_colwise_ptr=%p\n",
    (void*)amax_rowwise_ptr, (void*)amax_colwise_ptr
  );
  fflush(stderr);
#endif


  // Dispatch based on input dtype
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      alignas(64) CUtensorMap tensor_map_input{};
      alignas(64) CUtensorMap tensor_map_output{};
      alignas(64) CUtensorMap tensor_map_output_transpose{};

      create_2D_tensor_map(tensor_map_input,
                           input.data,
                           rows, cols,
                           BUFF_DIM_Y, BUFF_DIM_X,
                           cols, 0,
                           sizeof(IType) * 8);

      int element_bits =
#if MXFP4_SIMULATE_WITH_FP8
          8;
#else
          4;
#endif

      create_2D_tensor_map(tensor_map_output,
                           output->data,
                           rows, cols,
                           BUFF_DIM_Y, BUFF_DIM_X,
                           cols, 0, element_bits);

      if (return_transpose) {
        create_2D_tensor_map(tensor_map_output_transpose,
                             output->columnwise_data,
                             cols, rows,
                             BUFF_DIM_X, BUFF_DIM_Y,
                             rows, 0, element_bits);
      }

      // Shared memory size
      constexpr size_t buff_elems       = BUFF_DIM_Y * BUFF_DIM_X;
      constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

      constexpr size_t buff_size_aligned_in =
          DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType),
                            TMA_SHMEM_ALIGNMENT);
#if MXFP4_SIMULATE_WITH_FP8
      constexpr size_t buff_size_aligned_out =
          DIVUP_TO_MULTIPLE(buff_elems_total /* bytes */,
                            TMA_SHMEM_ALIGNMENT);
#else
      constexpr size_t buff_size_aligned_out =
          DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8,
                            TMA_SHMEM_ALIGNMENT);
#endif

      constexpr size_t buff_size_scales =
          (CHUNK_DIM_Y * CHUNK_DIM_X) / 32 * sizeof(mxfp4_scale_t);

      constexpr size_t in_mem                = buff_size_aligned_in;
      constexpr size_t out_data_mem          = buff_size_aligned_out;
      constexpr size_t out_data_transpose_mem = buff_size_aligned_out;
      constexpr size_t out_scales_transpose_mem = buff_size_scales;

      constexpr size_t out_mem =
          out_data_mem + out_data_transpose_mem;

      constexpr size_t dshmem_size =
          in_mem + out_mem + out_scales_transpose_mem +
          TMA_SHMEM_ALIGNMENT + 1024;

      // Kernel launch selection
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_transpose, RETURN_TRANSPOSE,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_global_scale, USE_GLOBAL_SCALE, 
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_encode_centric, ENCODE_CENTRIC, {
                    using KernelFn = void (*)(
                        const CUtensorMap,
                        const CUtensorMap,
                        const CUtensorMap,
                        mxfp4_scale_t *const,
                        mxfp4_scale_t *const,
                        const float *,
                        const float *const,
                        const float *const,
                        const size_t,
                        const size_t,
                        const size_t,
                        const size_t,
                        const size_t *);

                    KernelFn kernel;

                    if constexpr (use_2d_quantization) {
                      kernel =
                          &mxfp4_transpose::mxfp4_transpose_kernel_2D<
                              COMPUTE_ACTIVATIONS, ParamOP, OP,
                              IType, USE_STOCHASTIC_ROUNDING,
                              RETURN_TRANSPOSE, USE_GLOBAL_SCALE,ENCODE_CENTRIC>;
                    } else {
                      kernel =
                          &mxfp4_transpose::mxfp4_transpose_kernel<
                              COMPUTE_ACTIVATIONS, ParamOP, OP,
                              IType, USE_STOCHASTIC_ROUNDING,
                              RETURN_TRANSPOSE, USE_GLOBAL_SCALE,ENCODE_CENTRIC>;
                    }

                    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(dshmem_size)));

                    kernel<<<grid, block_size, dshmem_size, stream>>>(
                        tensor_map_input,
                        tensor_map_output,
                        tensor_map_output_transpose,
                        scales_ptr,
                        scales_transpose_ptr,
                        noop_ptr,
                        amax_rowwise_ptr,
                        amax_colwise_ptr,
                        rows,
                        cols,
                        scale_stride,
                        scale_stride_transpose,
                        rng_state);
                  }))));  // USE_GLOBAL_SCALE / RETURN_TRANSPOSE / USE_STOCHASTIC_ROUNDING
      ); // TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_MXFP4_TRANSPOSE_CUH_
