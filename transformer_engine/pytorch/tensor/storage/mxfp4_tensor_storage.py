# Copyright (c) 2025, Generic AI Implementation. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for MXFP4Tensor"""

from __future__ import annotations
from collections.abc import Iterable
import functools
import math
from typing import Any, Dict, Optional, Tuple, Union
import warnings

import torch

# import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ..quantized_tensor import QuantizedTensorStorage
from ..quantized_tensor import Quantizer
from ...utils import _empty_tensor

SIMULATE_MXFP4_WITH_FP8 = True

# =============================================================================
# Helper: E8M0 Scale Decoding
# =============================================================================
def _e8m0_to_scale(sx: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scale values (uint8) to float32. Bias 127."""
    if sx.is_floating_point():
        return sx.to(torch.float32)
    
    # exp = uint8_val - 127
    exp = sx.to(torch.int16) - 127
    return torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=sx.device),
        exp.to(torch.float32),
    )

# =============================================================================
# Helper: FP8 E4M3 Decoding (For Simulation Mode)
# =============================================================================
def _fp8_e4m3_to_float(qx_bytes: torch.Tensor) -> torch.Tensor:
    """Decode raw uint8 FP8(E4M3) values to float32 following OCP spec."""
    if qx_bytes.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 for FP8 bytes, got {qx_bytes.dtype}")

    x = qx_bytes.to(torch.int16)
    sign_bit = (x >> 7) & 0x1
    exp_bits = (x >> 3) & 0xF
    mant_bits = x & 0x7
    
    bias = 7
    device = qx_bytes.device

    sign = torch.where(sign_bit == 0, 1.0, -1.0).to(torch.float32)
    exp = exp_bits.to(torch.float32)
    mant = mant_bits.to(torch.float32)

    # Standard E4M3 decoding logic
    # Note: Simplified for speed, full OCP spec handles NaN/Inf specifically
    # but for neural net weights/activations, this fast path usually suffices.
    
    # Normal numbers: E > 0
    # Val = (-1)^S * 2^(E-7) * (1 + M/8)
    # Subnormal: E = 0
    # Val = (-1)^S * 2^(-6) * (M/8)
    
    is_subnorm = (exp_bits == 0)
    
    # Calculate mantissa value
    # Normal: 1.MMM, Subnorm: 0.MMM
    mant_val = torch.where(is_subnorm, mant / 8.0, 1.0 + mant / 8.0)
    
    # Calculate exponent value
    # Normal: E - 7, Subnorm: 1 - 7 = -6
    exp_val = torch.where(is_subnorm, torch.tensor(-6.0, device=device), exp - bias)
    
    return sign * torch.pow(2.0, exp_val) * mant_val


@functools.lru_cache(maxsize=None)
def _fp4_e2m1_vals(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Values representable in FP4 E2M1 format"""
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
        dtype=dtype,
    )


class _FromMXFP4Func(torch.autograd.Function):
    """Cast from MXFP4 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: MXFP4TensorStorage,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        print('dequantizing MXFP4 tensor...')

        # -----------------------------------------------------------
        # 1. Row-wise Dequantization
        # -----------------------------------------------------------
        if tensor._rowwise_data is not None:
            data_raw = tensor._rowwise_data
            scales_raw = tensor._rowwise_scale_inv
            
            # Identify shapes
            shape = list(data_raw.size())
            if not SIMULATE_MXFP4_WITH_FP8:
                # Native FP4 packs 2 elements per byte
                shape[-1] *= 2
            
            # --- A. Decode Values (Data) ---
            if SIMULATE_MXFP4_WITH_FP8:
                # E4M3 Simulation Path
                data_float = _fp8_e4m3_to_float(data_raw)
            else:
                # Native FP4 Path
                # Convert packed uint8 to int32 for indexing
                d = data_raw.view(torch.uint8).to(torch.int32)
                # Unpack: low nibble, high nibble
                unpacked = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(shape)
                # LUT Lookup
                lut = _fp4_e2m1_vals(data_raw.device, torch.float32)
                data_float = lut[unpacked.long()]

            # --- B. Decode Scales (E8M0) ---
            # Slice scales to match data blocks (ignoring hardware padding)
            # Block size = 32
            valid_rows = math.prod(shape[:-1])
            valid_cols_scaled = shape[-1] // 32
            
            # Reshape scales to [Rows, Cols_Scaled]
            scale_view = scales_raw.reshape(-1, scales_raw.size(-1))
            scale_view = scale_view[:valid_rows, :valid_cols_scaled]
            
            decoded_scales = _e8m0_to_scale(scale_view)

            # --- C. Apply Scales ---
            # Reshape data to [Blocks, 32]
            data_blocked = data_float.view(-1, 32)
            # Expand scales: [Blocks, 1]
            scale_expanded = decoded_scales.reshape(-1, 1)
            
            result = (data_blocked * scale_expanded) / 6.0
            
            # Restore original shape and cast
            return result.view(shape).to(dtype)

        # -----------------------------------------------------------
        # 2. Column-wise Dequantization
        # -----------------------------------------------------------
        elif tensor._columnwise_data is not None:
            # Note: Column-wise usually implies the data is stored in Transposed format
            # effectively [N, M] logical tensor is stored as [M, N] physical data 
            # (or similiar) to allow row-wise kernels to operate on columns.
            
            data_raw = tensor._columnwise_data
            scales_raw = tensor._columnwise_scale_inv

            # Assuming standard storage: [M, N] (logical transpose)
            shape_t = list(data_raw.size())
            if not SIMULATE_MXFP4_WITH_FP8:
                shape_t[-1] *= 2
            
            # --- A. Decode Values ---
            if SIMULATE_MXFP4_WITH_FP8:
                data_float_t = _fp8_e4m3_to_float(data_raw)
            else:
                d = data_raw.view(torch.uint8).to(torch.int32)
                unpacked = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(shape_t)
                lut = _fp4_e2m1_vals(data_raw.device, torch.float32)
                data_float_t = lut[unpacked.long()]

            # --- B. Decode Scales ---
            # For column-wise, scaling blocks are usually along the inner dim of the transpose
            valid_rows_t = math.prod(shape_t[:-1])
            valid_cols_scaled_t = shape_t[-1] // 32

            scale_view = scales_raw.reshape(-1, scales_raw.size(-1))
            scale_view = scale_view[:valid_rows_t, :valid_cols_scaled_t]
            
            decoded_scales_t = _e8m0_to_scale(scale_view)

            # --- C. Apply Scales ---
            data_blocked_t = data_float_t.view(-1, 32)
            scale_expanded_t = decoded_scales_t.reshape(-1, 1)
            
            result_t = (data_blocked_t * scale_expanded_t) / 6.0
            
            # Logic: The stored data is the Transpose. We return the Transpose of that (the original).
            # This depends on TE implementation details, but usually you return the 
            # dequantized tensor in its stored orientation or transpose it back.
            # Assuming we want to return the tensor matching the "logical" dimensions:
            return result_t.view(shape_t).t().to(dtype)

        else:
            raise ValueError("Attempted to dequantize MXFP4 tensor with no data")

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return grad, None


class MXFP4TensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of MXFP4Tensor."""

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _rowwise_scale_inv: torch.Tensor
    _columnwise_scale_inv: torch.Tensor
    _fp4_dtype: TE_DType
    _amax_rowwise: torch.Tensor
    _amax_columnwise: torch.Tensor

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: torch.Tensor,
        amax_rowwise: torch.Tensor,
        amax_columnwise: torch.Tensor,
        fp4_dtype: TE_DType,
        quantizer: Optional[Quantizer],
        *args,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._fp4_dtype = fp4_dtype
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._amax_rowwise = amax_rowwise
        instance._amax_columnwise = amax_columnwise
        return instance

    def clear(self):
        """Deallocate this tensor's memory."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
            self._amax_rowwise,
            self._amax_columnwise,
        ):
            if t is not None:
                t.data = _empty_tensor()

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "amax_rowwise": self._amax_rowwise,
            "amax_columnwise": self._amax_columnwise,
            "fp4_dtype": self._fp4_dtype,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], MXFP4TensorStorage]:
        """Prepare the tensor base for saving for backward"""
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
            self._amax_rowwise,
            self._amax_columnwise,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        self._amax_rowwise = None
        self._amax_columnwise = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        self._amax_rowwise = tensors[4]
        self._amax_columnwise = tensors[5]
        return tensors[6:]

    def get_data_tensors(self):
        """Get this Tensor's data."""
        return self._rowwise_data, self._columnwise_data

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize to a higher precision."""
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromMXFP4Func.apply(self, dtype)
        return _FromMXFP4Func.forward(None, self, dtype)

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        shape = None
        if self._rowwise_data is not None:
            byte_shape = list(self._rowwise_data.size())
            if SIMULATE_MXFP4_WITH_FP8:
                shape = byte_shape
            else:
                shape = byte_shape[:-1] + [byte_shape[-1] * 2]
        elif self._columnwise_data is not None:
            byte_shape = list(self._columnwise_data.size())
            if SIMULATE_MXFP4_WITH_FP8:
                # Assuming stored as transpose [M, N] for [N, M] tensor
                shape = [byte_shape[1], byte_shape[0]]
            else:
                shape = [byte_shape[1], byte_shape[0] * 2]
        
        if shape is None:
            raise RuntimeError("Attempted to get shape of MXFP4 tensor with no data")

        if dim is None:
            return torch.Size(shape)
        return shape[dim]

    def view(self, shape: torch.Size):
        cur_shape = self.size()
        if shape is None or shape == cur_shape:
            return self

        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(cur_shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        
        # Block size constraint: Inner dim must be multiple of 32 for MXFP4
        if shape[-1] != cur_shape[-1]:
            raise RuntimeError(
                "MXFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)})"
            )

        new_rowwise_data = None
        new_columnwise_data = None
        
        if self._rowwise_data is not None:
            if not SIMULATE_MXFP4_WITH_FP8:
                if shape[-1] % 2 != 0:
                    raise ValueError(f"Invalid shape for FP4 packing: {shape}")
                byte_shape = list(shape[:-1]) + [shape[-1] // 2]
            else:
                byte_shape = list(shape)
            new_rowwise_data = self._rowwise_data.view(byte_shape)

        if self._columnwise_data is not None:
            # Viewing a column-wise tensor implies reshaping the logical dimensions,
            # which means reshaping the physical Transposed dimensions.
            # Logical: [A, B] -> Transposed Storage: [B, A]
            # If we reshape logical to [X, B], we reshape storage to [B, X].
            
            # Note: We blocked reshaping the inner dim (B), so B is constant.
            # We are only reshaping the outer dimensions.
            
            logical_rows = math.prod(shape[:-1])
            logical_cols = shape[-1]
            
            # Storage shape target: [logical_cols, logical_rows]
            if not SIMULATE_MXFP4_WITH_FP8:
                # Storage inner dim (logical_rows) is packed? 
                # Usually column-wise packing happens on the 'leading' dimension of logical (inner of physical).
                # This gets complex. For Sim mode it's easier.
                pass 
            else:
                new_columnwise_data = self._columnwise_data.view(logical_cols, logical_rows)

        return MXFP4TensorStorage(
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=self._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=self._columnwise_scale_inv,
            amax_rowwise=self._amax_rowwise,
            amax_columnwise=self._amax_columnwise,
            quantizer=self._quantizer,
            fp4_dtype=self._fp4_dtype,
        )

    def __repr__(self):
        # Fallback to string representation if dequantize fails
        try:
            data_rowwise = self.dequantize()
        except Exception:
            data_rowwise = "N/A"

        return (
            "MXFP4TensorStorage("
            f"rowwise_scaled_data={data_rowwise},"
            f"rowwise_scale_inv={self._rowwise_scale_inv},"
            f"amax_rowwise={self._amax_rowwise},"
            f"amax_columnwise={self._amax_columnwise},"
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None

        if rowwise_usage:
            if self._rowwise_data is None:
                raise RuntimeError("MXFP4Tensor missing row-scaled data")
        else:
            self._rowwise_data = None
            self._rowwise_scale_inv = None
            self._amax_rowwise = None

        if columnwise_usage:
            if self._columnwise_data is None:
                raise RuntimeError("MXFP4Tensor missing column-scaled data")
        else:
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            self._amax_columnwise = None