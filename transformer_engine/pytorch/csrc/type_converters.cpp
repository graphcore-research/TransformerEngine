/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>

#include "common.h"
#include "pybind.h"

#ifndef MXFP4_SIMULATE_WITH_FP8
#define MXFP4_SIMULATE_WITH_FP8 1
#endif

namespace transformer_engine::pytorch {
namespace detail {

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(quantizer->get_scaling_mode());

  bool data_exists = !tensor.attr("_data").is_none();
  bool transpose_exists =
      !tensor.attr("_transpose_invalid").cast<bool>() && !tensor.attr("_transpose").is_none();

  NVTE_CHECK(data_exists || transpose_exists, "No data found for FP8 Tensor.");

  // FP8 data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (data_exists) {
    const auto &data = tensor.attr("_data").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
  }

  // FP8 data transpose
  if (transpose_exists) {
    const auto &data_transpose = tensor.attr("_transpose").cast<at::Tensor>();
    ret.set_columnwise_data(data_transpose.data_ptr(), fp8_dtype, getTensorShape(data_transpose));
  }

  // Scale-inverse
  {
    const auto &scale_inv = tensor.attr("_scale_inv").cast<at::Tensor>();
    float *dptr = reinterpret_cast<float *>(scale_inv.data_ptr());
    const auto &dtype = GetTransformerEngineDType(scale_inv.scalar_type());
    const auto &shape = getTensorShape(scale_inv);
    ret.set_rowwise_scale_inv(dptr, dtype, shape);
    ret.set_columnwise_scale_inv(dptr, dtype, shape);
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromMXFP8Tensor(py::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(NVTE_MXFP8_1D_SCALING);

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for MXFP8 Tensor.");

  // Row-scaled data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (rowwise_usage) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    ret.set_columnwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 getTensorShape(scale_inv));
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromFloat8BlockwiseQTensor(py::handle tensor, Quantizer *quantizer) {
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();
  bool is_2D_scaled = tensor.attr("_is_2D_scaled").cast<bool>();

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  auto ret = TensorWrapper(is_2D_scaled ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);

  if (rowwise_usage) {
    const at::Tensor &data_rowwise = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const at::Tensor &scale_inv_rowwise = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_rowwise_dptr = scale_inv_rowwise.data_ptr();
    const auto &rowwise_shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, rowwise_shape);
    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr, DType::kFloat32, scale_inv_rowwise_shape);
  }
  if (columnwise_usage) {
    const at::Tensor &data_colwise = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const at::Tensor &scale_inv_colwise = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    void *scale_inv_colwise_dptr = scale_inv_colwise.data_ptr();
    const auto &shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);

    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr, DType::kFloat32, scale_inv_colwise_shape);
  }
  quantizer->set_quantization_params(&ret);
  return ret;
}

TensorWrapper NVTETensorFromNVFP4Tensor(py::handle tensor, Quantizer *quantizer) {
  const DType dtype = tensor.attr("_fp4_dtype").cast<DType>();

  auto ret = TensorWrapper(NVTE_NVFP4_1D_SCALING);

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for NVFP4 Tensor.");

  // Row-scaled data
  if (rowwise_usage) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    const auto &amax_rowwise = tensor.attr("_amax_rowwise").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), dtype,
                         convert_shape_back_from_fp4(getTensorShape(data), false));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, getTensorShape(scale_inv));
    ret.set_amax(amax_rowwise.data_ptr(), DType::kFloat32, getTensorShape(amax_rowwise));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    const auto &amax_columnwise = tensor.attr("_amax_columnwise").cast<at::Tensor>();
    ret.set_columnwise_data(data.data_ptr(), DType::kFloat4E2M1,
                            convert_shape_back_from_fp4(getTensorShape(data), false));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3,
                                 getTensorShape(scale_inv));
    ret.set_columnwise_amax(amax_columnwise.data_ptr(), DType::kFloat32,
                            getTensorShape(amax_columnwise));
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

// [FIX] Added MXFP4 Implementation
// [DEBUG] Instrumented MXFP4 Implementation
TensorWrapper NVTETensorFromMXFP4Tensor(py::handle tensor, Quantizer *quantizer) {
  // fprintf(stderr, "\n[DEBUG] >>> Entering NVTETensorFromMXFP4Tensor\n");

  const DType dtype = tensor.attr("_fp4_dtype").cast<DType>();
  auto ret = TensorWrapper(NVTE_MXFP4_1D_SCALING);

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  // fprintf(stderr, "[DEBUG] Usage: Rowwise=%d, Columnwise=%d\n", rowwise_usage, columnwise_usage);

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for MXFP4 Tensor.");

  // Helper to get or create a safe float32 tensor (scalar 1.0)
  auto get_safe_scalar = [&](const char* attr_name, const char* internal_name) -> at::Tensor {
      if (py::hasattr(tensor, attr_name) && !tensor.attr(attr_name).is_none()) {
          auto t = tensor.attr(attr_name).cast<at::Tensor>();
          // fprintf(stderr, "[DEBUG] Found '%s' from Python. Ptr: %p\n", attr_name, t.data_ptr());
          return t;
      }
      // Check if we already created a backup
      if (py::hasattr(tensor, internal_name) && !tensor.attr(internal_name).is_none()) {
          auto t = tensor.attr(internal_name).cast<at::Tensor>();
          // fprintf(stderr, "[DEBUG] Using cached internal '%s'. Ptr: %p\n", internal_name, t.data_ptr());
          return t;
      }
      // Create new 1.0 tensor
      // fprintf(stderr, "[DEBUG] Creating NEW internal '%s' (fallback to 1.0).\n", internal_name);
      auto opts = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
      at::Tensor safe_tensor = at::ones({1}, opts);
      tensor.attr(internal_name) = safe_tensor; 
      // fprintf(stderr, "[DEBUG] New internal ptr: %p\n", safe_tensor.data_ptr());
      return safe_tensor;
  };

  // 1. Global Scale
  at::Tensor global_scale = get_safe_scalar("_scale", "_internal_global_scale");
  ret.set_scale(global_scale.data_ptr(), DType::kFloat32, getTensorShape(global_scale));
  // fprintf(stderr, "[DEBUG] ret.set_scale called with %p\n", global_scale.data_ptr());

  // 2. Prepare Safe AMAX (1.0)
  // This looks for _amax_rowwise first, effectively treating it as the "main" amax
  at::Tensor primary_amax = get_safe_scalar("_amax_rowwise", "_internal_amax_rowwise");

  // 3. Row-scaled data
  if (rowwise_usage) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    
    // Explicitly set the ROWWISE amax
    ret.set_amax(primary_amax.data_ptr(), DType::kFloat32, getTensorShape(primary_amax));
    // fprintf(stderr, "[DEBUG] ret.set_amax (rowwise) called with %p\n", primary_amax.data_ptr());

#if MXFP4_SIMULATE_WITH_FP8
    ret.set_rowwise_data(data.data_ptr(), dtype, getTensorShape(data));
#else
    ret.set_rowwise_data(data.data_ptr(), dtype,
                         convert_shape_back_from_fp4(getTensorShape(data), false));
#endif
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
  }

  // 4. Column-scaled data
  if (columnwise_usage) {
    const auto &data = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    
    // Check specific column amax
    at::Tensor col_amax;
    if (py::hasattr(tensor, "_amax_columnwise") && !tensor.attr("_amax_columnwise").is_none()) {
        col_amax = tensor.attr("_amax_columnwise").cast<at::Tensor>();
        // fprintf(stderr, "[DEBUG] Found '_amax_columnwise' from Python. Ptr: %p\n", col_amax.data_ptr());
    } else {
        col_amax = primary_amax; 
        // fprintf(stderr, "[DEBUG] '_amax_columnwise' missing. Using primary fallback. Ptr: %p\n", col_amax.data_ptr());
    }
    
    ret.set_columnwise_amax(col_amax.data_ptr(), DType::kFloat32, getTensorShape(col_amax));
    // fprintf(stderr, "[DEBUG] ret.set_columnwise_amax called with %p\n", col_amax.data_ptr());

#if MXFP4_SIMULATE_WITH_FP8
    ret.set_columnwise_data(data.data_ptr(), dtype, getTensorShape(data));
#else
    ret.set_columnwise_data(data.data_ptr(), dtype,
                            convert_shape_back_from_fp4(getTensorShape(data), false));
#endif
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 getTensorShape(scale_inv));
  } else {
    // CRITICAL: Even if no column data, populate column-amax just in case GEMM logic (transa=N) requests it.
    // fprintf(stderr, "[DEBUG] Column usage FALSE. Force-setting columnwise_amax to primary (%p) for safety.\n", primary_amax.data_ptr());
    ret.set_columnwise_amax(primary_amax.data_ptr(), DType::kFloat32, getTensorShape(primary_amax));
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);
  
  // fprintf(stderr, "[DEBUG] <<< Exiting NVTETensorFromMXFP4Tensor\n\n");
  return ret;
}

}  // namespace detail

}  // namespace transformer_engine::pytorch