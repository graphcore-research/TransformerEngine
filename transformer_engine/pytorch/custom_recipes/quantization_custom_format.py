# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Custom FP4 quantization recipe reference implementation."""

import dataclasses
import os
import sys
from typing import Optional, Tuple, Union, Any

import torch
import torch.utils._pytree as pytree
import gfloat
import functools

from transformer_engine.pytorch.custom_recipes import quantization
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
import re
from gfloat.formats import format_info_ocp_e2m1,format_info_ocp_e4m3,format_info_ocp_e8m0,format_info_ocp_e5m2, format_info_bfloat16
from gfloat.types import FormatInfo,Domain
from low_bits_training.quantization.customTorchQuantiser import round_ndarray
from transformer_engine.pytorch.custom_recipes.quantization_custom_kernel import _compiled_quantize_core

format_info_ocp_e8m3 = FormatInfo(
    name="ocp_e8m3",
    k=12,
    precision=4,
    bias=2 ** (8 - 1) - 1,
    has_nz=True,
    num_high_nans=0,
    domain=Domain.Finite,

    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)


format_info_ocp_ue5m3 = FormatInfo(
    name="ocp_ue5m3",
    k=8,
    precision=4,
    bias=15,
    has_nz=False,
    domain=Domain.Finite,

    num_high_nans=0,
    has_subnormals=True,
    is_signed=False,
    is_twos_complement=False,
)

# --- IMPORTS FROM CUSTOM PATH ---
# We try to import the custom quantization logic.
# The user specified paths in /workspace/low-bits-training/low_bits_training/quantization/
# We assume the package structure allows 'from low_bits_training.quantization ...'
try:
    from low_bits_training.quantization.dimensionQuantisationClass import _core_scaling_math
    # We might need other things like extraction of E/M bits if we want to be fully robust
    # But _core_scaling_math handles most logic.
except ImportError:
    print("[CustomQuantizer] Warning: Could not import low_bits_training.quantization. Falling back to internal logic where possible or failing.")
    _core_scaling_math = None

LOOKUP = {
                     "E4M3": gfloat.formats.format_info_ocp_e4m3,
                     "E5M2": gfloat.formats.format_info_ocp_e5m2,
                     "E8M0": gfloat.formats.format_info_ocp_e8m0,
                     "E5M3": format_info_ocp_ue5m3,
                     "E2M1": gfloat.formats.format_info_ocp_e2m1,
                     "MXFP4": gfloat.formats.format_info_ocp_e8m0, # S_b is E8M0
                     "NVFP4": gfloat.formats.format_info_ocp_e8m0, # If forced?
                 }

def custom_ref_rht_2d_quantizer_factory(role):
    """
    Quantizer factory for Custom FP4 recipe reference implementation (RHT and 2D quantization for weights).
    
    Reads environment variables:
      NVTE_CUSTOM_DISABLE_RHT: Set to "1" to disable RHT.
      NVTE_CUSTOM_SCALE_FORMAT: Scaling format (e.g. "E8M0", "E4M3", "MXFP4"). Default "E8M0".
      NVTE_CUSTOM_BLOCK_SIZE: Block size (e.g. 16, 32). Default 32.
    """
    disable_rht = os.getenv("NVTE_CUSTOM_DISABLE_RHT", "0") == "1"
    with_rht = not disable_rht
    
    # Custom Configurations
    scale_format = os.getenv("NVTE_CUSTOM_SCALE_FORMAT", "E5M3")
    block_size = int(os.getenv("NVTE_CUSTOM_BLOCK_SIZE", "32"))
    use_global_scale = os.getenv("NVTE_CUSTOM_USE_GLOBAL_SCALE", "1") == "1"
    encode_centric = os.getenv("NVTE_CUSTOM_ENCODE_CENTRIC", "0") == "1"
    # Determine tile shape from block size
    # TE expects (outer, inner). For block quantization we usually care about inner.
    # MXFP4 uses (1, 32). NVFP4 uses (1, 16) or (16, 16).
    quant_tile_shape = (1, block_size)

    if role == "linear_input":
        return CustomQuantizerRef(
            dtype=utils.Fp4Formats.E2M1,
            quant_tile_shape=quant_tile_shape,
            scale_format=scale_format,
            with_rht=with_rht,
            pow_2_scales=False,
            use_global_scale=use_global_scale,
            encode_centric=encode_centric,
        )
    if role == "linear_weight":
        disable_2d = os.getenv("NVTE_CUSTOM_DISABLE_2D_QUANTIZATION", "0") == "1"
        # If 2D is enabled and block size is 16, we might use (16, 16).
        # But for custom block sizes, we usually default to 1D blocks unless specified.
        # We will stick to 1D blocks for simplicity unless explicitly requested or matching NVFP4 2D.
        # If block_size is 16 and not disabled 2d, we use (16,16) to match NVFP4 default?
        # User asked to "freely chose block size".
        if not disable_2d:
             w_tile_shape = (block_size, block_size)
        else:
             w_tile_shape = quant_tile_shape

        return CustomQuantizerRef(
            dtype=utils.Fp4Formats.E2M1,
            quant_tile_shape=w_tile_shape,
            scale_format=scale_format,
            with_rht=False,
            pow_2_scales=False,
            use_global_scale=use_global_scale,
            encode_centric=encode_centric,
        )
    if role in ("linear_grad_output", "linear_grad", "linear_grad_input"):
        return CustomQuantizerRef(
            dtype=utils.Fp4Formats.E2M1,
            quant_tile_shape=quant_tile_shape,
            scale_format=scale_format,
            with_rht=with_rht,
            pow_2_scales=False,
            use_global_scale=use_global_scale,
            encode_centric=encode_centric,
        )
    return None


def get_custom_quantizer_factory(
    scale_format="E8M0",
    block_size=32,
    use_global_scale=True,
    encode_centric=False,
    with_rht=False,
    scale_round_mode="TiesToEven",
    round_mode="TiesToEven",
    with_2d_weights=False,
    eps=0.0,
    with_random_sign_mask=True,
):
    """
    Quantizer factory for Custom FP4 recipe reference implementation.
    """
    
    # Enable RHT if requested
    
    # Custom Configurations
    
    # Determine tile shape from block size
    quant_tile_shape = (1, block_size)

    def factory(role):
        if role == "linear_input":
            return CustomQuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=quant_tile_shape,
                scale_format=scale_format,
                with_rht=with_rht,
                pow_2_scales=False,
                use_global_scale=use_global_scale,
                encode_centric=encode_centric,
                scale_round_mode=scale_round_mode,
                round_mode=round_mode,
                eps=eps,
                with_random_sign_mask=with_random_sign_mask
            )
        if role == "linear_weight":
            # If 2D quantization is enabled, use square blocks
            if with_2d_weights:
                 w_tile_shape = (block_size, block_size)
            else:
                 w_tile_shape = quant_tile_shape

            return CustomQuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=w_tile_shape,
                scale_format=scale_format,
                with_rht=False,
                pow_2_scales=False,
                use_global_scale=use_global_scale,
                encode_centric=encode_centric,
                scale_round_mode=scale_round_mode,
                round_mode=round_mode,
                eps=eps,
                with_random_sign_mask=with_random_sign_mask
            )
        if role in ("linear_grad_output", "linear_grad", "linear_grad_input"):
            return CustomQuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=quant_tile_shape,
                scale_format=scale_format,
                with_rht=with_rht,
                pow_2_scales=False,
                use_global_scale=use_global_scale,
                encode_centric=encode_centric,
                scale_round_mode=scale_round_mode,
                round_mode=round_mode,
                eps=eps,
                with_random_sign_mask=with_random_sign_mask
            )
        return None
    return factory 

def cast_to_fp4x2(x):
    """Quantize a tensor to FP4 E2M1 and store in a byte tensor"""

    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


def pack_fp4_values(x: torch.Tensor) -> torch.Tensor:
    """Pack rounded FP4 E2M1 values (floats) into uint8 tensor."""
    # Assuming x contains values exactly present in E2M1 dynamic range
    
    # Handle sign (bit 3)
    sign = torch.signbit(x)
    abs_x = torch.abs(x)
    
    # Init indices
    indices = torch.zeros_like(x, dtype=torch.uint8)
    
    # Map magnitude to 0-7 using robust thresholds
    indices[(abs_x > 0.25) & (abs_x < 0.75)] = 1
    indices[(abs_x >= 0.75) & (abs_x <= 1.25)] = 2
    indices[(abs_x > 1.25) & (abs_x < 1.75)] = 3
    indices[(abs_x >= 1.75) & (abs_x <= 2.5)] = 4
    indices[(abs_x > 2.5) & (abs_x < 3.5)] = 5
    indices[(abs_x >= 3.5) & (abs_x <= 5.0)] = 6
    indices[abs_x > 5.0] = 7
    
    # Add sign bit
    indices += (sign.to(torch.uint8) * 8)
    
    # Pack columns (low nibble = even cols, high nibble = odd cols)
    return indices[:, ::2] + indices[:, 1::2] * 16


def cast_from_fp4x2(x, dq_dtype):
    """Dequantize FP4 E2M1 tensor that has been represented in a byte tensor"""
    fp4_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=x.device,
        dtype=dq_dtype,
    )

    # Convert to long integers for indexing
    second_bit = torch.div(x, 16, rounding_mode="floor").to(torch.long)
    first_bit = (x - second_bit * 16).to(torch.long)

    # Use the long integers to index fp4_values
    first_bit_values = fp4_values[first_bit]
    second_bit_values = fp4_values[second_bit]

    result = torch.zeros(
        (first_bit_values.shape[0], first_bit_values.shape[1] * 2),
        device=x.device,
        dtype=dq_dtype,
    )
    result[:, ::2] = first_bit_values
    result[:, 1::2] = second_bit_values

    return result


def cast_to_e4m3(decode_scale, global_amax):
    """Scale and cast to FP8 E4M3."""
    decode_scale = decode_scale * global_amax
    FLOAT8_E4M3_MAX = torch.tensor(448.0, device=decode_scale.device, dtype=torch.float32)
    decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
    return decode_scale.to(torch.float8_e4m3fn)


def high_precision_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    accumulate: bool = False,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_alpha: float = 1.0,
) -> torch.Tensor:
    """GEMM implementation with unquantized data"""
    # Handle transpositions
    mat1, mat2 = a, b
    if is_a_transposed:
        mat1 = a.T
    if is_b_transposed:
        mat2 = b.T

    # Ensure dtype compatibility for torch.addmm
    mat1 = mat1.to(out_dtype)
    mat2 = mat2.to(out_dtype)

    # Determine output shape
    y_shape = (mat1.size(0), mat2.size(1))

    if bias is not None:
        bias = bias.to(out_dtype)
        # With bias case
        if out_dtype == torch.float32:
            y_ref = torch.addmm(bias.repeat(mat1.size(0), 1), mat1, mat2, beta=1, alpha=1)
        else:
            y_ref = torch.addmm(bias, mat1, mat2, beta=1, alpha=scale_alpha)
    else:
        # Without bias case
        if accumulate and out is not None:
            y_ref = out.clone().to(out_dtype)
        else:
            y_ref = torch.zeros(y_shape, dtype=out_dtype, device=a.device)
        torch.addmm(y_ref, mat1, mat2, beta=1, alpha=scale_alpha, out=y_ref)

    return y_ref


@dataclasses.dataclass
class CustomTensorRef(QuantizedTensorStorage):
    """Custom FP4 tensor for middleware between Transformer Engine and Kitchen."""

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    data_t: Optional[torch.Tensor] = None
    scale_t: Optional[torch.Tensor] = None
    global_amax_row: Optional[torch.Tensor] = None
    global_amax_col: Optional[torch.Tensor] = None

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    quant_dtype: Optional[Union[utils.Fp4Formats, torch.dtype]] = None
    original_shape: Optional[Tuple[int, ...]] = None
    _quantizer: Optional[Quantizer] = None

    @property
    def experimental(self) -> bool:
        return True

    def get_tensor(self, rowwise: bool) -> torch.Tensor:
        return self.data if rowwise else self.data_t

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to high precision."""
        if dtype is None:
            dtype = self.dtype
        if self.data is None:
            return None
        
        
        # Dequantize unpacked bits to [-6, 6] range values
        if self.data.dtype.is_floating_point:
             out = self.data
        else:
             out = cast_from_fp4x2(self.data, torch.float32)
        
        # Apply scales
        M, K = out.shape
        # Get block length from quantizer if available, or infer
        block_length = self._quantizer.quant_tile_shape[1]
             
        sx = self.scale.to(torch.float32)
        
        gA = self.global_amax_row.to(torch.float32)
        
        fmt = self._quantizer.fmt
        
        SCALE_MAX = float(fmt.max)
        DATA_MAX = 6.0 # Fixed for E2M1
        
        factor = SCALE_MAX * DATA_MAX
        
        # Optimization: Broadcast scales instead of repeat_interleave
        if self._quantizer.use_global_scale:
             # Combine sx (M, G) with gA (M, 1) or scalar
             combined_sx = sx * (gA / factor)
        else:
             combined_sx = sx
             
        # Broadcast (M, G) -> (M, G, 1)
        sx_view = combined_sx.unsqueeze(-1)
        
        # View Out: (M, K) -> (M, K//block, block)
        out_view = out.view(M, -1, block_length)
        
        # Apply Scale
        out = (out_view * sx_view).reshape(M, K)


                 
        return out.to(dtype)

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], QuantizedTensorStorage]:
        tensors = [self.data, self.data_t, self.scale, self.scale_t]
        self.data = None
        self.data_t = None
        self.scale = None
        self.scale_t = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        self.data = tensors[0]
        self.data_t = tensors[1]
        self.scale = tensors[2]
        self.scale_t = tensors[3]
        return tensors[4:]

    # Compatibility
    @property
    def _data(self):
        return self.data

    @_data.setter
    def _data(self, value):
        self.data = value

    @property
    def _scale_inv(self):
        return self.scale

    @_scale_inv.setter
    def _scale_inv(self, value):
        self.scale = value

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"quant_dtype={self.quant_dtype}, "
            f"original_shape={self.original_shape}"
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        has_data = self.data is not None
        has_data_transpose = self.data_t is not None
        needs_data = has_data
        needs_data_transpose = has_data_transpose

        if rowwise_usage is not None:
            needs_data = rowwise_usage
        if columnwise_usage is not None:
            needs_data_transpose = columnwise_usage

        if needs_data and not has_data:
            raise RuntimeError("Cannot generate FP4 data, even from FP4 data transpose")
        if needs_data_transpose and not has_data_transpose:
            if not has_data:
                raise RuntimeError("FP4 data is required to generate FP4 data transpose")
            self._create_transpose()

        if not needs_data:
            self.data = None
        if not needs_data_transpose:
            self.data_t = None

    def _create_transpose(self):
        if not self.data.is_contiguous():
            self.data = self.data.contiguous()
        self.data_t = self.data.t().contiguous()
        self.scale_t = self.scale

    def size(self, *args, **kwargs):
        assert self.original_shape is not None
        return torch.Size(self.original_shape)


def get_wgrad_sign_vector() -> torch.Tensor:
    return torch.tensor(
        [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
        dtype=torch.float32,
    )


class CustomQuantizerRef(Quantizer):
    """Custom FP4 quantizer for middleware between Transformer Engine and Kitchen"""

    def __init__(
        self,
        dtype: utils.Fp4Formats,
        rowwise: bool = True,
        columnwise: bool = True,
        pow_2_scales: bool = False, # Deprecated in Custom, but kept for signature compat
        use_global_scale: bool = False,
        eps: float = 0.0,
        quant_tile_shape: Tuple[int, int] = (1, 16),
        with_rht: bool = False,
        with_random_sign_mask: bool = True,
        encode_centric: bool = False,
        scale_format: str = "E8M0", # NEW: Format of the scale (e.g. E8M0, E4M3)
        scale_round_mode: str = "TiesToEven",
        round_mode: str = "TiesToEven",
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True
        self.is_reference_quantizer = True

        self.dtype = dtype
        self.pow_2_scales = pow_2_scales 
        self.use_global_scale = use_global_scale
        self.eps = eps
        self.quant_tile_shape = quant_tile_shape
        self.with_rht = with_rht
        self.with_random_sign_mask = with_random_sign_mask
        self.encode_centric = encode_centric
        self.scale_format = scale_format
        self.scale_round_mode = scale_round_mode
        self.round_mode = round_mode
        
        # Pre-resolve FormatInfo to avoid global lookups during graph capture
        self.fmt = LOOKUP.get(scale_format, LOOKUP["E8M0"])

    @property
    def experimental(self) -> bool:
        return True

    @staticmethod
    @functools.lru_cache(maxsize=16)
    def _build_hadamard_matrix_cached(
        size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Construct a Hadamard matrix (cached)"""
        h = torch.ones((1, 1), device=device, dtype=torch.float32)
        while h.shape[0] < size:
            h = torch.cat(
                [
                    torch.cat([h, h], dim=1),
                    torch.cat([h, -h], dim=1),
                ],
                dim=0,
            )
        return h.to(dtype)

    def _apply_rht(self, x: torch.Tensor) -> torch.Tensor:
        if not self.with_rht:
            return x

        rht_dim = self.quant_tile_shape[1]
        
        original_shape = x.shape

        # Get cached raw Hadamard matrix (no signs)
        H = self._build_hadamard_matrix_cached(rht_dim, x.device, x.dtype)
        scale = 1.0 / float(rht_dim) ** 0.5

        if self.with_random_sign_mask:
            signs = get_wgrad_sign_vector().to(x.device).to(x.dtype)
            if rht_dim > signs.shape[0]:
                repeats = (rht_dim + signs.shape[0] - 1) // signs.shape[0]
                signs = signs.repeat(repeats)[:rht_dim]
            elif rht_dim < signs.shape[0]:
                signs = signs[:rht_dim]
            
            x_reshaped = x.contiguous().view(-1, rht_dim)
            x_signed = x_reshaped * signs
            x_mat = x_signed
        else:
            x_mat = x.contiguous().view(-1, rht_dim)
        transform = H * scale
        out = x_mat @ transform
        return out.view(original_shape)

    @staticmethod
    def _recover_swizzled_scales(
        swizzled_scale: bool, scale: torch.Tensor, m: int, n: int, block_length: int
    ) -> torch.Tensor:
        if not swizzled_scale:
            return scale
        rounded_m = utils.roundup_div(m, 128) * 128
        scale_n = utils.roundup_div(n, block_length)
        rounded_n = utils.roundup_div(scale_n, 4) * 4
        tmp = torch.reshape(scale, (rounded_m // 128, rounded_n // 4, 32, 4, 4))
        tmp = torch.permute(tmp, (0, 3, 2, 1, 4))
        result = torch.reshape(tmp, (rounded_m, rounded_n))
        return result[:m, :scale_n]

    @classmethod
    def _quantize_blockwise_reference(
        cls,
        x: torch.Tensor,
        global_amax: torch.Tensor,
        tile_len_x: int,
        tile_len_y: int,
        *,
        pow_2_scales: bool,
        eps: float,  
        use_global_scale: bool = True,
        scale_format: str = "E8M0", # Passed from instance
        encode_centric: bool = False, # Passed from instance
        scale_round_mode_str: str = "TiesToEven",
        round_mode_str: str = "TiesToEven",
        fmt: Optional[Any] = None, # Pass FormatInfo object directly
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compile-Friendly Optimization: Delegate to standalone kernel
        # This ensures the quantization subgraph is fused by Dynamo

        
        # Data Format is E2M1 (Fixed as per verified logic)
        data_fmt = format_info_ocp_e2m1

        m, n = x.shape
        using_2d_quantization = tile_len_x == tile_len_y
        
        if fmt is None:
             raise ValueError("FormatInfo 'fmt' is required for compiled kernel execution")

        return _compiled_quantize_core(
            x,
            global_amax,
            tile_len_x,
            tile_len_y,
            using_2d_quantization,
            float(fmt.max), # Scale MAX
            float(data_fmt.max),  # Data MAX
            encode_centric,
            use_global_scale,
            scale_round_mode_str,
            round_mode_str,
            # Data Format Primitives (E2M1)
            float(data_fmt.max),
            float(data_fmt.min),
            int(data_fmt.precision),
            int(data_fmt.bias),
            bool(data_fmt.has_subnormals),
            bool(data_fmt.is_signed),
            bool(data_fmt.has_nz),
            int(data_fmt.k),
            bool(getattr(data_fmt, 'is_twos_complement', False)),
            
            # Scale Format Primitives (fmt)
            float(fmt.max),
            float(fmt.min),
            int(fmt.precision),
            int(fmt.bias),
            bool(fmt.has_subnormals),
            bool(fmt.is_signed),
            bool(fmt.has_nz),
            int(fmt.k),
            bool(getattr(fmt, 'is_twos_complement', False)),
            
            # Defaults
            int(getattr(data_fmt, 'num_high_nans', 0)),
            int(getattr(fmt, 'num_high_nans', 0))
        )

             # Hande 2D Quantization (Square Blocks)
             

        # --- FALLBACK TO ORIGINAL NVFP4 LOGIC ---
        # 1. Calculate Block Maxima (vec_max)
        

    @staticmethod
    def _pad_tensor(
        tensor: torch.Tensor, row_divisor: Optional[int], col_divisor: Optional[int]
    ) -> torch.Tensor:

        M, N = tensor.shape
        padding_needed_rows = 0
        padding_needed_cols = 0

        if row_divisor is not None and M % row_divisor != 0:
            padding_needed_rows = row_divisor - (M % row_divisor)
        # Check and calculate column padding if col_divisor is provided
        if col_divisor is not None and N % col_divisor != 0:
            padding_needed_cols = col_divisor - (N % col_divisor)

        # Return original tensor if no padding is needed
        if padding_needed_rows == 0 and padding_needed_cols == 0:
            return tensor

        # pad the tensor
        out = torch.nn.functional.pad(
            tensor,
            (0, padding_needed_cols, 0, padding_needed_rows),
            mode="constant",
            value=0.0,
        ).contiguous()

        return out

    @staticmethod
    def _rm_pad_tensor(tensor: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:

        M, N = original_size
        out = tensor[:M, :N].contiguous()
        return out

    def _quantize(self, tensor: torch.Tensor) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
# 1. Prepare inputs common to both MXFP4 and NVFP4
        # Row-input will always be the original input.
        row_input = tensor
        col_input = (
            self._apply_rht(tensor.t().contiguous())
            if self.with_rht
            else tensor.t().contiguous()
        )

        global_amax_row = torch.max(torch.abs(row_input)).to(torch.float32).view(1)
        global_amax_col = (
            torch.max(torch.abs(col_input)).to(torch.float32).view(1)
            if self.columnwise_usage
            else global_amax_row
        )

        transpose_scales = False

        M, N = tensor.shape
        if self.rowwise_usage:
            x_input = row_input
            x_padded = self._pad_tensor(
                x_input, row_divisor=self.quant_tile_shape[0], col_divisor=self.quant_tile_shape[1]
            )

            qx, sx = self._quantize_blockwise_reference(
                x_padded,
                global_amax_row,
                self.quant_tile_shape[1],
                self.quant_tile_shape[0],
                pow_2_scales=self.pow_2_scales,
                eps=self.eps,
                use_global_scale=self.use_global_scale, 
                scale_format=self.scale_format, # Pass format
                encode_centric=self.encode_centric,
                scale_round_mode_str=self.scale_round_mode,
                round_mode_str=self.round_mode,
                fmt=self.fmt,
            )
            if transpose_scales:
                sx = sx.T

            # Dynamic slicing based on whether we packed or not
            if qx.dtype == torch.uint8:
                qx = self._rm_pad_tensor(qx, (M, N // 2))
            else:
                qx = self._rm_pad_tensor(qx, (M, N))

        else:
            qx = None
            sx = None

        if self.columnwise_usage:
            x_t = col_input
            x_t_padded = self._pad_tensor(
                x_t, row_divisor=self.quant_tile_shape[0], col_divisor=self.quant_tile_shape[1]
            )

            qx_t, sx_t = self._quantize_blockwise_reference(
                x_t_padded,
                global_amax_col,
                self.quant_tile_shape[1],
                self.quant_tile_shape[0],
                pow_2_scales=self.pow_2_scales,
                eps=self.eps,
                use_global_scale=self.use_global_scale,
                scale_format=self.scale_format,
                encode_centric=self.encode_centric,
                scale_round_mode_str=self.scale_round_mode,
                round_mode_str=self.round_mode,
                fmt=self.fmt,
            )

            if qx_t.dtype == torch.uint8:
                qx_t = self._rm_pad_tensor(qx_t, (N, M // 2))
            else:
                qx_t = self._rm_pad_tensor(qx_t, (N, M))

            if transpose_scales:
                sx_t = sx_t.T
        else:
            qx_t = None
            sx_t = None

        return qx, sx, qx_t, sx_t, global_amax_row, global_amax_col

    def quantize(
        self,
        tensor: torch.Tensor,
        **kwargs, 
    ) -> CustomTensorRef:

        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.shape[-1])

        qx, sx, qx_t, sx_t, global_amax_row, global_amax_col = self._quantize(tensor)

        return CustomTensorRef(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            global_amax_row=global_amax_row,
            global_amax_col=global_amax_col,
            dtype=tensor.dtype,
            device=tensor.device,
            quant_dtype=self.dtype,
            _quantizer=self,
            original_shape=original_shape,
        )

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: Any,  # MMParams
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        accumulate: bool = False,
        gemm_type: Any = None, # GEMMType
        qresult_x: Optional[CustomTensorRef] = None,
        qresult_w: Optional[CustomTensorRef] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Python implementation of microblock FP4 GEMM (Mimicking NVFP4 Ref)."""
        from transformer_engine.pytorch.custom_recipes.quantization import GEMMType
        if qx.dtype == torch.uint8:
            high_precision_x = cast_from_fp4x2(qx, out_dtype)
            high_precision_w = cast_from_fp4x2(qw, out_dtype)
        else:
            high_precision_x = qx.to(out_dtype)
            high_precision_w = qw.to(out_dtype)

        sx = sx.to(torch.float32)
        sw = sw.to(torch.float32)
        
        SCALE_MAX = float(self.fmt.max)
        DATA_MAX = 6.0
        
        factor = (SCALE_MAX * DATA_MAX) ** 2.0

        if gemm_type == GEMMType.WGRAD:
            # For WGRAD check use_global_scale
            if self.use_global_scale:
                partial_alpha = qresult_x.global_amax_col * qresult_w.global_amax_col
                alpha = torch.div(partial_alpha, factor).squeeze(-1)
            else:
                 # No global scaling factor
                 alpha = torch.tensor(1.0, device=qx.device, dtype=out_dtype)
        else:
            if self.use_global_scale:
                partial_alpha = qresult_x.global_amax_row * qresult_w.global_amax_row
                alpha = torch.div(partial_alpha, factor).squeeze(-1)
            else:
                 alpha = torch.tensor(1.0, device=qx.device, dtype=out_dtype)

        M, K = high_precision_x.shape
        N, K_w = high_precision_w.shape

        # Determine block length from quantizer or guess
        block_length = self.quant_tile_shape[1]
             
        # Expand scales to match input shapes
        # sx: (M, grid_k) -> (M, K)
        # BROADCASTING OPTIMIZATION: Avoid repeat_interleave materialization
        
        # x: (M, K) -> (M, K//block, block)
        x_view = high_precision_x.view(M, -1, block_length)
        # sx: (M, K//block) -> (M, K//block, 1)
        sx_view = sx.unsqueeze(-1)
        x_scaled = (x_view * sx_view).reshape(M, K)
        
        # w: (N, K) -> (N, K//block, block)
        w_view = high_precision_w.view(N, -1, block_length)
        # sw: (N, K//block) -> (N, K//block, 1)
        sw_view = sw.unsqueeze(-1)
        w_scaled = (w_view * sw_view).reshape(N, K)

        # Perform GEMM
        # y = x_scaled @ w_scaled.T
        y = high_precision_gemm_ref(
            x_scaled, w_scaled, torch.float32, is_b_transposed=True
        )

        if K > 0 and self.use_global_scale:
            # only apply global scale for non-empty cases
            y = alpha * y

        # accumulation happens at epilogue in float32
        if accumulate:
            y += out.to(torch.float32)

        if bias is not None:
            y += bias.view(1, -1).to(torch.float32)

        return y.to(out_dtype)

def _custom_tensor_ref_flatten(c):
    # Return (list of tensors), (auxiliary context context)
    return [c.data, c.scale, c.data_t, c.scale_t, c.global_amax_row, c.global_amax_col], (c.dtype, c.device, c.quant_dtype, c.original_shape, c._quantizer)

def _custom_tensor_ref_unflatten(values, context):
    dtype, device, quant_dtype, original_shape, _quantizer = context
    data, scale, data_t, scale_t, global_amax_row, global_amax_col = values
    return CustomTensorRef(
        data=data, scale=scale, data_t=data_t, scale_t=scale_t,
        global_amax_row=global_amax_row, global_amax_col=global_amax_col,
        dtype=dtype, device=device, quant_dtype=quant_dtype,
        _quantizer=_quantizer, original_shape=original_shape
    )

# Register it!
pytree.register_pytree_node(CustomTensorRef, _custom_tensor_ref_flatten, _custom_tensor_ref_unflatten)