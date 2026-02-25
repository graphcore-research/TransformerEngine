/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file core_mxfp4.cuh
 *  \brief Core functions used in MXFP4 quantization.
 *  Adapted from mxfp4_transpose.cuh and core_nvfp4.cuh.
 *
 *  MXFP4 uses E8M0 block scales with block size 32, unlike NVFP4 which
 *  uses E4M3 scales with block size 16 and a global FP32 scale.
 */

#ifndef TRANSFORMER_ENGINE_CORE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_CORE_MXFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <limits>

#include "../../common.h"
#include "../../util/curanddx.hpp"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif  // FP4_TYPE_SUPPORTED

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {

// MXFP4 scales are E8M0 (uint8), block size is 32
using mxfp4_scale_t = e8m0_t;
constexpr size_t MXFP4_BLOCK_SIZE = 32;

namespace core {

#if FP4_TYPE_SUPPORTED
using namespace ptx;

// -------------------------------------------------------
// E8M0 exponent helpers
// -------------------------------------------------------

/// Returns 2^(b - 127), i.e. the decode scale value for an E8M0 byte.
__device__ __forceinline__ float exp2f_e8m0(uint8_t b) {
  int e = (int)b - 127;
  if (e > 127) return FLT_MAX;  // b==255 -> overflow
  return ldexpf(1.0f, e);
}

/// Returns 2^(127 - b) = 1 / exp2f_e8m0(b), i.e. the reciprocal decode scale.
__device__ __forceinline__ float exp2f_rcp_e8m0(uint8_t b) {
  return ldexpf(1.0f, 127 - (int)b);
}

// -------------------------------------------------------
// Scale computation helpers
// -------------------------------------------------------

/// Compute the E8M0 decode-centric scaling factor.
/// Returns biased exponent: round(log2(block_amax / fp4_max)) + 127
/// This is the standard MX convention: stored scale S such that
/// dequantized_value = fp4_value * 2^(S - 127).
__device__ __forceinline__ mxfp4_scale_t compute_decoding_scaling_factor(
    const float block_amax) {
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f

  if (block_amax <= 1.0e-9f) {
    return static_cast<mxfp4_scale_t>(0);  // exponent = -127
  }

  const float effective = block_amax / fp4_max;
  float exponent_f = roundf(log2f(effective));
  int exponent_i = (int)exponent_f;
  exponent_i = max(-127, min(128, exponent_i));
  return static_cast<mxfp4_scale_t>(exponent_i + 127);
}

/// Compute the E8M0 encode-centric scaling factor (reciprocal).
/// Returns biased exponent: round(log2(fp4_max / block_amax)) + 127
__device__ __forceinline__ mxfp4_scale_t compute_encoding_scaling_factor(
    const float block_amax) {
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f

  if (block_amax <= 1.0e-9f) {
    return static_cast<mxfp4_scale_t>(255);  // max E8M0
  }

  const float effective = fp4_max / block_amax;
  float exponent_f = roundf(log2f(effective));
  int exponent_i = (int)exponent_f;
  exponent_i = max(-127, min(128, exponent_i));
  return static_cast<mxfp4_scale_t>(exponent_i + 127);
}

// -------------------------------------------------------
// RNG helper
// -------------------------------------------------------

using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                     curanddx::PhiloxRounds<10>() +
                     curanddx::SM<800>() +
                     curanddx::Thread());

__device__ __forceinline__ uint32_t get_rbits(RNG &rng, uint4 &random_uint4,
                                               int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    curanddx::uniform_bits dist;
    random_uint4 = dist.generate4(rng);
  }
  const uint32_t *const rbits_arr = reinterpret_cast<uint32_t *>(&random_uint4);
  return rbits_arr[rnd_idx++];
}

// -------------------------------------------------------
// FP4 conversion helpers using PTX
// -------------------------------------------------------

/// Convert 4 BF16 values to FP4x4 with stochastic rounding.
/// The input is pre-multiplied by the inverse block scale.
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_sr(
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

/// Convert 4 BF16 values to FP4x4 with round-to-nearest.
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_rn(
    const uint64_t in_4x, const float2 scale, const uint32_t /*rbits*/) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  uint32_t out_4x = 0;
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
    return mul_cvt_bf16_to_fp4_4x_with_sr(in_4x, scale, rbits);
  } else {
    return mul_cvt_bf16_to_fp4_4x_with_rn(in_4x, scale, rbits);
  }
}

/// Convert 4 FP32 values to FP4x4 with stochastic rounding.
__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_sr(
    const float2 in01, const float2 in23, const float2 scale,
    const uint32_t rbits) {
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

/// Convert 4 FP32 values to FP4x4 with round-to-nearest.
__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_rn(
    const float2 in01, const float2 in23, const float2 scale,
    const uint32_t /*rbits*/) {
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
    const float2 in01, const float2 in23, const float2 scale,
    const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_fp32_to_fp4_4x_with_sr(in01, in23, scale, rbits);
  } else {
    return mul_cvt_fp32_to_fp4_4x_with_rn(in01, in23, scale, rbits);
  }
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CORE_MXFP4_CUH_
