import dataclasses
from typing import Optional, Tuple, Union

import torch

from transformer_engine.pytorch.custom_recipes import quantization
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
from .quantization_nvfp4 import NVFP4QuantizerRef,high_precision_gemm_ref,cast_from_fp4x2


def mxfp4_ref_quantizer_factory(role):
    """
    Quantizer factory for MXFP4 reference implementation.
    """
    import os
    encode_centric = os.getenv("NVTE_MXFP4_ENCODE_CENTRIC", "1") == "1"

    if role == "linear_input":
        return MXFP4QuantizerRef(
            encode_centric=encode_centric,
            quant_tile_shape=(1, 32),
            use_global_scale=True,
        )
    if role == "linear_weight":
        return MXFP4QuantizerRef(
            encode_centric=encode_centric,
            quant_tile_shape=(32, 32),
            use_global_scale=True,
        )
    if role in ("linear_grad_output", "linear_grad", "linear_grad_input"):
        return MXFP4QuantizerRef(
            encode_centric=encode_centric,
            quant_tile_shape=(1, 32),
            use_global_scale=True,
        )
    return None


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

def e8m0_to_scale(sx: torch.Tensor) -> torch.Tensor:
    """
    Decode E8M0 scale values to float32.

    E8M0 is exponent-only with bias 127 (see OCP MX spec).
    We treat raw value b in [0,255] as exponent e = b - 127 and scale = 2**e.

    If sx is already floating-point, we assume it's a decoded scale and just cast.
    """
    if sx.is_floating_point():
        return sx.to(torch.float32)

    if sx.dtype != torch.uint8:
        raise TypeError(f"Unexpected E8M0 dtype: {sx.dtype} (expected uint8 or float)")

    exp = sx.to(torch.int16) - 127 # exponent range [-127, 128]
    return torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=sx.device),
        exp.to(torch.float32),
    )

def fp8_e4m3_to_float(qx_bytes: torch.Tensor) -> torch.Tensor:
    """Decode raw uint8 FP8(E4M3) values to float32."""
    if qx_bytes.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 for FP8 bytes, got {qx_bytes.dtype}")

    x = qx_bytes.to(torch.int16)
    sign = torch.where((x >> 7) & 1 == 0, 1.0, -1.0).to(torch.float32)
    exp = ((x >> 3) & 0xF).to(torch.int16)
    mant = (x & 0x7).to(torch.float32)
    bias = 7

    out = torch.zeros_like(mant, dtype=torch.float32, device=qx_bytes.device)

    # Normal: E != 0
    is_normal = exp != 0
    # Subnorm: E == 0, M != 0
    is_subnorm = (exp == 0) & (mant != 0)
    # Zero: E == 0, M == 0 (already 0.0 in out)
    
    # Logic for NaN (E=15, M!=0) omitted for MXFP4 sim as valid inputs shouldn't produce NaNs

    if is_normal.any():
        e = (exp[is_normal] - bias).to(torch.float32)
        m = 1.0 + mant[is_normal] / 8.0
        out[is_normal] = sign[is_normal] * torch.pow(2.0, e) * m

    if is_subnorm.any():
        e = float(1 - bias)
        m = mant[is_subnorm] / 8.0
        out[is_subnorm] = sign[is_subnorm] * (2.0 ** e) * m

    return out

def get_wgrad_sign_vector() -> torch.Tensor:
    """Hard-coded signs for Hadamard transform"""
    return torch.tensor(
        [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
        dtype=torch.float32,
    )

class MXFP4QuantizerRef(NVFP4QuantizerRef):
    """
    MXFP4 Quantizer Reference Implementation.
    
    Supports both standard (Decode-Centric) and inverted (Encode-Centric) quantization strategies.
    """
    def __init__(
        self,
        dtype: utils.Fp4Formats = utils.Fp4Formats.E2M1,
        rowwise: bool = True,
        columnwise: bool = True,
        # Forced to True for MXFP4
        pow_2_scales: bool = True,
        use_global_scale: bool = False,
        # New Flag
        encode_centric: bool = False,
        quant_tile_shape: Tuple[int, int] = (1, 32),
        with_rht: bool = False, 
        with_random_sign_mask: bool = False, 
        eps: float = 0.0,
        simulate_mxfp4_with_fp8: bool = False,
    ):
        # Enforce MXFP4 requirement
        assert pow_2_scales is True, "MXFP4 requires power-of-2 scales (E8M0)"

        super().__init__(
            dtype=dtype,
            rowwise=rowwise,
            columnwise=columnwise,
            pow_2_scales=pow_2_scales, 
            use_global_scale=use_global_scale,
            eps=eps,
            quant_tile_shape=quant_tile_shape,
            with_rht=with_rht,
            with_random_sign_mask=with_random_sign_mask,
        )
        self.encode_centric = encode_centric
        self.simulate_mxfp4_with_fp8 = simulate_mxfp4_with_fp8

    def _quantize_blockwise_mxfp4(
        self,
        x: torch.Tensor,
        global_amax: torch.Tensor,
        tile_len_x: int,
        tile_len_y: int,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Specialized blockwise quantization for MXFP4.
        Handles both Decode-Centric (Standard) and Encode-Centric (Inverse) logic.
        """
        assert x.ndim == 2
        m, n = x.shape
        using_2d_quantization = tile_len_x == 32 and tile_len_y == 32
        
        # 1. Calculate Block Maxima (m_p)
        if using_2d_quantization:
            x_blocks = (
                x.unfold(0, tile_len_y, tile_len_y)
                .unfold(1, tile_len_x, tile_len_x)
                .to(torch.float32)
            )
            block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))
            # Broadcast back to match original shape's block layout
            vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)
        else:
            x_reshaped = x.view(m, n // tile_len_x, tile_len_x)
            vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(torch.float32)
            
        x_view = x.view(m, n // tile_len_x, tile_len_x)
        FLOAT4_E2M1_MAX = 6.0

        # 2. Determine Global Scaling Factor (S_enc)
        # S_enc = 6.0 / G
        if self.use_global_scale:
            safe_global = torch.where(global_amax <= 1e-9, torch.tensor(1.0, device=x.device), global_amax)
            
            # --- FIX START: Force float32 division and storage ---
            numerator = torch.tensor(FLOAT4_E2M1_MAX, device=x.device, dtype=torch.float32)
            s_enc = (numerator / safe_global.to(torch.float32)).to(torch.float32)
        else:
            # Absolute mode: S_enc = 6.0 so that effective_val = block_amax.
            # This ensures GEMM's hardcoded 1/36 factor works with absolute scales.
            s_enc = torch.tensor(6.0, device=x.device)

        # 3. Calculate Scale Indices (E8M0) based on strategy
        if self.encode_centric:
            # === ENCODE CENTRIC ===
            # Target: Multiplier S ~ 6.0 / (m_p * S_enc)
            # This is the "Inverse" logic.
            
            # effective_val = 6.0 / (vec_max * s_enc)
            denom = vec_max * s_enc
            effective_val = FLOAT4_E2M1_MAX / denom
            
            # Log2 + Round (not Ceil) to avoid saturation
            exponent = torch.round(torch.log2(effective_val))
            
            # Handle Zeros: If block is 0, multiplier is Infinite. Set to Max Scale.
            is_zero = vec_max <= 1e-9
            exponent = torch.where(is_zero, torch.tensor(128.0, device=x.device), exponent)
            
            # Clamp to E8M0
            exponent = torch.clamp(exponent, -127, 128)
            mult_bits = (exponent + 127).to(torch.uint8)
            
            # 3.5 Flip for storage (Hardware expects Divisor bits)
            # CUDA: 254 - mult_bits
            scale_indices = (254 - mult_bits.to(torch.int32)).clamp(0, 255).to(torch.uint8)
            
            # 4. Calculate Application Scale
            # We apply the stored multiplier directly.
            # Applied = S_enc * Stored
            stored_val = torch.pow(2.0, exponent)
            applied_scale = s_enc * stored_val

        else:
            # === DECODE CENTRIC (Standard) ===
            # Target: Divisor S ~ (m_p * S_enc) / 6.0
            
            effective_val = (vec_max * s_enc) / FLOAT4_E2M1_MAX
            # Log2 + Round (to match CUDA roundf)
            exponent = torch.round(torch.log2(effective_val))
            # Handle Zeros: If block is 0, divisor is 0. Set to Min Scale.
            is_zero = vec_max <= 1e-9
            exponent = torch.where(is_zero, torch.tensor(-127.0, device=x.device), exponent)
            
            # Clamp
            exponent = torch.clamp(exponent, -127, 128)
            scale_indices = (exponent + 127).to(torch.uint8)
            
            # 4. Calculate Application Scale
            # We apply the inverse of the stored divisor.
            # Applied = S_enc * (1 / Stored)
            stored_val = torch.pow(2.0, exponent)
            # print(f"[REF BLOCK 0] Exponent before zero handling: {exponent[0,0].item():.6f}")

            applied_scale = s_enc / stored_val

        # 5. Apply Scale to Data
        applied_scale = applied_scale.to(torch.float32)
        scaled_x = x_view.to(torch.float32) * applied_scale

        # 6. Clip and Cast (Mock FP4)
        clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)
        
        # Return casted data and the indices
        # Squeeze the last dim of indices to match expected (M, K/32) shape
        return cast_to_fp4x2(clipped_x), scale_indices.squeeze(-1)

    # Override the main quantize loop to call our specialized blockwise function
    def _quantize(self, tensor: torch.Tensor) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        # 1. Prepare inputs (RHT / Transpose)
        row_input = tensor
        col_input = (
            self._apply_rht(tensor.t().contiguous())
            if self.with_rht
            else tensor.t().contiguous()
        )

        # 2. Compute Global Amax
        global_amax_row = torch.max(torch.abs(row_input)).to(torch.float32).view(1)
        global_amax_col = (
            torch.max(torch.abs(col_input)).to(torch.float32).view(1)
            if self.columnwise_usage
            else global_amax_row
        )

        transpose_scales = False
        M, N = tensor.shape

        # 3. Rowwise Quantization
        if self.rowwise_usage:
            x_padded = self._pad_tensor(
                row_input, row_divisor=self.quant_tile_shape[0], col_divisor=self.quant_tile_shape[1]
            )
            
            # Call specialized MXFP4 function
            qx, sx = self._quantize_blockwise_mxfp4(
                x_padded,
                global_amax_row,
                self.quant_tile_shape[1],
                self.quant_tile_shape[0],
                eps=self.eps,
            )
            
            if transpose_scales:
                sx = sx.T
            qx = self._rm_pad_tensor(qx, (M, N // 2)) # Packed shape adjustment
        else:
            qx = None
            sx = None

        # 4. Columnwise Quantization
        if self.columnwise_usage:
            x_t_padded = self._pad_tensor(
                col_input, row_divisor=self.quant_tile_shape[0], col_divisor=self.quant_tile_shape[1]
            )

            qx_t, sx_t = self._quantize_blockwise_mxfp4(
                x_t_padded,
                global_amax_col,
                self.quant_tile_shape[1],
                self.quant_tile_shape[0],
                eps=self.eps,
            )

            qx_t = self._rm_pad_tensor(qx_t, (N, M // 2))
            if transpose_scales:
                sx_t = sx_t.T
        else:
            qx_t = None
            sx_t = None

        return qx, sx, qx_t, sx_t, global_amax_row, global_amax_col

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,  # pylint: disable=unused-argument
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,
        qresult_x: QuantizedTensorStorage | None = None,
        qresult_w: QuantizedTensorStorage | None = None,
    ) -> torch.Tensor:
        """MXFP4 GEMM with Encode/Decode-centric support."""
        assert bias is None, "Bias is implemented for FP4 GEMM."

        if self.simulate_mxfp4_with_fp8:
            # Decode FP8 to Float32
            high_precision_x = fp8_e4m3_to_float(qx)
            high_precision_w = fp8_e4m3_to_float(qw)
        else:
            high_precision_x = cast_from_fp4x2(qx, torch.bfloat16)
            high_precision_w = cast_from_fp4x2(qw, torch.bfloat16)

        # 1. Decode E8M0 -> Float
        sx_val = e8m0_to_scale(sx) if sx.dtype == torch.uint8 else sx.to(torch.float32)
        sw_val = e8m0_to_scale(sw) if sw.dtype == torch.uint8 else sw.to(torch.float32)

        sx_real = sx_val
        sw_real = sw_val

        # 3. Global Scale Factor
        # Standard: factor = 36.0 (since inputs were mapped to [-6, 6])
        factor = 36.0 
        
        if self.use_global_scale and qresult_x and qresult_w:
            if gemm_type == quantization.GEMMType.WGRAD:
                gA = qresult_x.global_amax_col
                gB = qresult_w.global_amax_col
            else:
                gA = qresult_x.global_amax_row
                gB = qresult_w.global_amax_row
            
            # Reconstruct absolute magnitude:
            alpha = (gA * gB / 36.0).to(torch.float32).squeeze(-1)
        else:
            # Absolute mode: s_enc = 1.0 => alpha = 1.0 / 36.0
            # (Matches CUDA behavior where 1/36 is hardcoded in mxfp4 recipe)
            alpha = torch.tensor(1.0 / 36.0, device=high_precision_x.device)

        M, K = high_precision_x.shape
        N, _ = high_precision_w.shape
        block_length = 32
        grid_k = K // block_length
        y = torch.zeros(M, N, dtype=torch.float32, device=qx.device)

        # Tiled GEMM
        for k in range(grid_k):
            k_start = k * block_length
            k_end = k_start + block_length

            qx_blk = high_precision_x[:, k_start:k_end]
            qw_blk = high_precision_w[:, k_start:k_end]
            
            sx_blk = sx_real[:, k]
            sw_blk = sw_real[:, k]

            # In both encode and decode centric, Blackwell hardware GEMM 
            # always treats the stored E8M0 scale as a multiplier (Standard OCP).
            # Quantization already accounts for inversion if encode_centric is True.
            scale_prod = torch.outer(sx_blk, sw_blk)
            
            gemm_blk = high_precision_gemm_ref(
                qx_blk, qw_blk, torch.float32, is_b_transposed=True
            )
            
            y += scale_prod * gemm_blk

        y = alpha * y

        if accumulate and out is not None:
            y += out.to(torch.float32)

        return y.to(out_dtype)
    @staticmethod
    def _build_hadamard_matrix(
        size: int, device: torch.device, dtype: torch.dtype, with_random_sign_mask: bool = True
    ) -> torch.Tensor:
        """Construct a Hadamard matrix of given power-of-two size with entries +-1.

        Uses Sylvester construction to avoid SciPy dependency.
        """
        assert (size & (size - 1)) == 0, "Hadamard size must be a power of two"
        h = torch.ones((1, 1), device=device, dtype=torch.float32)
        while h.shape[0] < size:
            h = torch.cat(
                [
                    torch.cat([h, h], dim=1),
                    torch.cat([h, -h], dim=1),
                ],
                dim=0,
            )
        if with_random_sign_mask:
            sign_mat = get_wgrad_sign_vector().to(device) * torch.eye(
                size, device=device, dtype=torch.float32
            )
            h = sign_mat @ h
        return h.to(dtype)

    def _apply_rht(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomized Hadamard transform without random signs (reference path).

        This matches the reference used in tests: x_reshaped @ (H * (1/sqrt(g))).
        """
        # Only apply when enabled
        if not self.with_rht:
            return x

        # RHT dimension equals the quantization tile length (NVFP4 uses 16)
        rht_dim = 16
        assert (
            x.shape[-1] % rht_dim == 0
        ), f"Inner dimension {x.shape[-1]} must be divisible by hadamard dimension {rht_dim}"

        # Build H and scale
        H = self._build_hadamard_matrix(rht_dim, x.device, x.dtype, self.with_random_sign_mask)
        scale = 1.0 / float(rht_dim) ** 0.5

        # Perform blockwise transform along the last dimension
        original_shape = x.shape
        x_mat = x.contiguous().view(-1, rht_dim)
        # Random sign matrix is identity in this reference (no sign flipping)
        transform = H * scale
        out = x_mat @ transform
        return out.view(original_shape)
