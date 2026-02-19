# Copyright (c) 2025, Graphcore / Generic AI.
# Adapted from NVIDIA NVFP4 tests.

import pytest
import torch
import math
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import MXFP4QuantizerRef
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import MXFP4Quantizer, MXFP4Tensor # Your Custom Class
import os
# Detect whether this TE build is using the MXFP4_SIMULATE_WITH_FP8 path.
# You can set this before running pytest:
#   export NVTE_MXFP4_SIMULATE_WITH_FP8=1
SIMULATE_MXFP4_WITH_FP8 = True

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

    exp = sx.to(torch.int16) - 127  # exponent range [-127, 128]
    return torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=sx.device),
        exp.to(torch.float32),
    )


def fp8_e4m3_to_float(qx_bytes: torch.Tensor) -> torch.Tensor:
    """
    Decode raw uint8 FP8(E4M3) values to float32.

    Layout: [S][E3 E2 E1 E0][M2 M1 M0]
    - Sign bit:    bit 7
    - Exponent:    bits 6..3 (4 bits), bias = 7
    - Mantissa:    bits 2..0 (3 bits)

    We implement the OCP FP8 E4M3 semantics:
      - normal:   v = (-1)^S * 2^(E - bias) * (1 + M / 2^3)
      - subnorm:  v = (-1)^S * 2^(1 - bias) * (M / 2^3)
      - zero:     M == 0 and E == 0 -> 0
      - NaN:      E == 0xF and M != 0 -> NaN  (not really expected here)
    """
    if qx_bytes.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 for FP8 bytes, got {qx_bytes.dtype}")

    x = qx_bytes.to(torch.int16)

    sign_bit = (x >> 7) & 0x1
    exp_bits = (x >> 3) & 0xF
    mant_bits = x & 0x7

    bias = 7
    device = qx_bytes.device

    sign = torch.where(sign_bit == 0, 1.0, -1.0).to(torch.float32)

    exp = exp_bits.to(torch.int16)
    mant = mant_bits.to(torch.float32)

    is_zero = (exp == 0) & (mant_bits == 0)
    is_subnorm = (exp == 0) & (mant_bits != 0)
    is_nan = (exp_bits == 0xF) & (mant_bits != 0)
    is_normal = (~is_zero) & (~is_subnorm) & (~is_nan)

    out = torch.zeros_like(mant, dtype=torch.float32, device=device)

    # normal: v = sign * 2^(E - bias) * (1 + M / 2^3)
    if is_normal.any():
        exp_norm = (exp[is_normal] - bias).to(torch.float32)
        mant_norm = 1.0 + mant[is_normal] / 8.0  # 2^-3 = 1/8
        out[is_normal] = (
            sign[is_normal]
            * torch.pow(torch.tensor(2.0, device=device), exp_norm)
            * mant_norm
        )

    # subnorm: v = sign * 2^(1 - bias) * (M / 2^3)
    if is_subnorm.any():
        exp_sub = float(1 - bias)  # constant = -6 for E4M3
        mant_sub = mant[is_subnorm] / 8.0
        out[is_subnorm] = (
            sign[is_subnorm]
            * (2.0 ** exp_sub)
            * mant_sub
        )

    # zeros already set to 0
    if is_nan.any():
        out[is_nan] = float("nan")

    return out


# =============================================================================
# 2. Helper Functions
# =============================================================================

def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated

# =============================================================================
# Helper: Unpack FP4 (E2M1)
# =============================================================================
def unpack_fp4_to_float(x_packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks uint8 (2x FP4) indices into their float representations.
    Returns: (Values, Indices)
    """
    # 1. Unpack indices
    # Assumes standard packing: Low nibble = even index, High nibble = odd index
    flat = x_packed.flatten()
    low = flat & 0x0F
    high = (flat >> 4) & 0x0F
    
    # Interleave low/high to restore original order
    # stack: (N, 2) -> flatten: (2N)
    indices_flat = torch.stack((low, high), dim=1).flatten()
    
    # Restore 2D shape if input was 2D
    if x_packed.dim() == 2:
        indices = indices_flat.view(x_packed.shape[0], -1)
    else:
        indices = indices_flat.view(1, -1) # Default to 1 row if flattened input

    # 2. Map indices to values (E2M1) for readability
    #                  0    1    2    3    4    5    6    7
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x_packed.device)
    
    # Indices 8-15 are negative versions of 0-7
    signs = (indices >= 8).float()
    magnitude_indices = (indices % 8).long()
    values = lut[magnitude_indices]
    values = values * (1.0 - 2.0 * signs) # Apply sign
    
    return values, indices

# =============================================================================
# Debug Test
# =============================================================================
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# def test_mxfp4_deep_debug(dtype):
#     device = "cuda"
#     M, N = 128, 128
    
#     # 1. Create a Linear Ramp Input
#     # This makes it obvious if data is just shuffled vs broken.
#     x = torch.arange(M * N, dtype=dtype, device=device).reshape(M, N)
#     x = (x / (M * N)) * 4.0 
    
#     # Force the last block to have a known max value
#     # Block size 32. Last block starts at col 96.
#     x[-1, -32:] = 6.0 

#     print(f"\n\n{'='*40}")
#     print(f"DEBUG REPORT (M={M}, N={N}, {dtype})")
#     print(f"{'='*40}")

#     # 2. Reference Run
#     ref_q = MXFP4QuantizerRef().quantize(x)
#     ref_scale = ref_q.scale
#     ref_data_float, ref_indices = unpack_fp4_to_float(ref_q.data)

#     # 3. SUT Run
#     sut_q = MXFP4Quantizer(
#         fp4_dtype=tex.DType.kFloat8E4M3, 
#         rowwise=True, columnwise=False, with_rht=False
#     ).quantize(x)
    
#     sut_scale = sut_q._rowwise_scale_inv
#     sut_data_float, sut_indices = unpack_fp4_to_float(sut_q._rowwise_data)

#     # =========================================================================
#     # DIAGNOSTICS
#     # =========================================================================

#     # A. Scale Analysis
#     # -----------------
#     print(f"Shapes:")
#     print(f"  > REF Scale: {ref_scale.shape}")
#     print(f"  > SUT Scale: {sut_scale.shape}")

#     # Helper to get the last element safely regardless of shape
#     def get_last(t):
#         return t.flatten()[-1].item()

#     sut_scale_last = get_last(sut_scale)
#     ref_scale_last = get_last(ref_scale)

#     print(f"\nLast Block Scale (Input=6.0): Expected ~127")
#     print(f"  > REF Scale: {ref_scale_last}")
#     print(f"  > SUT Scale: {sut_scale_last}")

#     if sut_scale_last == 0:
#         print("\n[CRITICAL FAIL] SUT Scale is 0.")
#         print("Diagnosis: The Kernel computed AMAX=0 for this block.")
#         print("Root Cause: Likely TMA Descriptor misconfiguration or Shared Memory barrier race.")
#     elif sut_scale_last == ref_scale_last:
#          print("[PASS] Scales match for last block.")
#     else:
#          print(f"[FAIL] Scale mismatch. Diff: {sut_scale_last - ref_scale_last}")

#     # B. Data Saturation Check
#     # ------------------------
#     # Count how many saturated values (Index 7 or 15 => +/- 6.0)
#     num_sat_ref = (ref_indices % 8 == 7).sum().item()
#     num_sat_sut = (sut_indices % 8 == 7).sum().item()
    
#     print(f"\nSaturation Check (Count of +/- 6.0 output):")
#     print(f"  > REF Saturated Count: {num_sat_ref}")
#     print(f"  > SUT Saturated Count: {num_sat_sut}")
    
#     if num_sat_sut > num_sat_ref * 10:
#          print("[CRITICAL FAIL] SUT is heavily saturated. Inverse scale is likely massive.")

#     # C. Swizzle Check (Middle Block)
#     # -------------------------------
#     # Compare indices for a middle block
#     mid_idx = M // 2
    
#     # Robust slicing
#     def slice_block(t, row, start_col, width=32):
#         if t.dim() == 2: return t[row, start_col:start_col+width]
#         # Fallback for flattened
#         flat_idx = row * N + start_col
#         return t.flatten()[flat_idx:flat_idx+width]

#     sut_block = slice_block(sut_indices, mid_idx, 0)
#     ref_block = slice_block(ref_indices, mid_idx, 0)

#     print(f"\nMiddle Block Data Sample (Sorted):")
#     sut_sorted, _ = torch.sort(sut_block)
#     ref_sorted, _ = torch.sort(ref_block)
def check_quantization_mxfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
    global_scaling: bool,
    with_2d_quantization: bool,
    encode=bool
) -> None:
    device = "cuda"
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 1. Generate Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # 2. Pad Input for SUT (C++ requires multiple of 32)
    ALIGN = 32
    pad_rows = (ALIGN - (M % ALIGN)) % ALIGN
    pad_cols = (ALIGN - (N % ALIGN)) % ALIGN
    
    if pad_rows > 0 or pad_cols > 0:
        x_sut_input = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0.0)
    else:
        x_sut_input = x

    M_pad, N_pad = x_sut_input.shape

    # 3. Run SUT (CUDA)
    if SIMULATE_MXFP4_WITH_FP8:
        alloc_dtype = tex.DType.kFloat8E4M3 
    else:
        alloc_dtype = tex.DType.kFloat4E2M1

    mxfp4_quantizer = MXFP4Quantizer(
        fp4_dtype=alloc_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_rht=False,
        global_scaling=global_scaling,
        with_2d_quantization=with_2d_quantization,
        encode_centric=encode
    )

    if use_cpp_allocator:
        x_mxfp4_sut = mxfp4_quantizer.quantize(x_sut_input)
    else:
        x_mxfp4_sut = mxfp4_quantizer.make_empty(
            (M_pad, N_pad), dtype=x_dtype, device=device, requires_grad=False
        )
        x_mxfp4_sut = mxfp4_quantizer.update_quantized(x_sut_input, x_mxfp4_sut)

    # 4. Extract & Prepare SUT Outputs
    # ----------------------------------------------------
    qx_sut_raw = x_mxfp4_sut._rowwise_data.view(dtype=torch.uint8) 
    sx_sut_raw = x_mxfp4_sut._rowwise_scale_inv

    # Slice Data to logical (M, N)
    qx_sut = qx_sut_raw[:M, :N].contiguous()

    # --- Scale Extraction Logic ---
    scale_cols = math.ceil(N / 32)
    
    if with_2d_quantization:
        # SUT Behavior: Returns (M_pad, N_pad/32).
        # In 2D mode, the kernel writes the SAME scale to 32 consecutive rows.
        # We must stride to get unique block scales.
        # Shape becomes (ceil(M/32), ceil(N/32))
        sx_sut_unique = sx_sut_raw[0:M_pad:32, :scale_cols].contiguous()
    else:
        # 1D Behavior: One scale per row.
        # Shape becomes (M, ceil(N/32))
        sx_sut_unique = sx_sut_raw[:M, :scale_cols].contiguous()

    # 5. Run Reference
    # ----------------------------------------------------
    # Ref handles unpadded inputs natively
    ref_quantizer = MXFP4QuantizerRef(
        columnwise=return_transpose,
        use_global_scale=global_scaling,
        quant_tile_shape=(32, 32) if with_2d_quantization else (1, 32),
        pow_2_scales=True,
        simulate_mxfp4_with_fp8=False,
        encode_centric=encode
    )
    x_mxfp4_ref = ref_quantizer.quantize(x)

    qx_ref = x_mxfp4_ref.data          
    sx_ref = x_mxfp4_ref.scale         


    # Ensure ref scale shape is consistent (Ref might output 320-based blocks if it padded internally)
    # We clip Ref scales to match the SUT unique shape logic
    rows_unique = sx_sut_unique.shape[0]
    cols_unique = sx_sut_unique.shape[1]
    if with_2d_quantization:
        # For 2D quantization, the reference also replicates scales across 32 rows.
        # We need to stride by 32 to get the unique block scales, just like SUT.
        sx_ref = sx_ref[0:M:32, :cols_unique]
    else:
        sx_ref = sx_ref[:rows_unique, :cols_unique]


    # 6. Dequantize & Compare
    # ----------------------------------------------------
    
    # SUT Expansion
    scales_sut = e8m0_to_scale(sx_sut_unique)
    
    if with_2d_quantization:
        # Expand (BlocksY, BlocksX) -> (M_pad, N_pad)
        # Note: We must expand to full padded size first to match strides, then slice.
        # 1. Expand Cols: (BlocksY, N_pad)
        s_exp = scales_sut.repeat_interleave(32, dim=1)
        # 2. Expand Rows: (M_pad, N_pad)
        s_exp = s_exp.repeat_interleave(32, dim=0)
        # 3. Slice to logical size
        scales_sut_expanded = s_exp[:M, :N]
    else:
        # Expand (M, BlocksX) -> (M, N)
        s_exp = scales_sut.repeat_interleave(32, dim=1)
        scales_sut_expanded = s_exp[:M, :N]

    # Ref Expansion
    scales_ref = e8m0_to_scale(sx_ref)
    if with_2d_quantization:
        s_ref_exp = scales_ref.repeat_interleave(32, dim=1).repeat_interleave(32, dim=0)
        scales_ref_expanded = s_ref_exp[:M, :N]
    else:
        s_ref_exp = scales_ref.repeat_interleave(32, dim=1)
        scales_ref_expanded = s_ref_exp[:M, :N]

    # --- Float Comparison ---
    if SIMULATE_MXFP4_WITH_FP8:
        p_vals_sut = fp8_e4m3_to_float(qx_sut)
    else:
        p_vals_sut, _ = unpack_fp4_to_float(qx_sut)

    ref_vals, _ = unpack_fp4_to_float(qx_ref)

    # In both encode and decode centric mode, dequantization treats 
    # the stored E8M0 scale as a magnitude (Divisor), but we must 
    # normalize by fp4_max (6.0) because S_enc=6.0 was used.
    x_hat_sut = (p_vals_sut * scales_sut_expanded / 6.0).to(torch.float32)
    x_hat_ref = (ref_vals * scales_ref_expanded / 6.0).to(torch.float32)

    # --- Debug ---
    abs_err = (x_hat_sut - x_hat_ref).abs()
    max_abs = abs_err.max().item()
    print(f"MXFP4(sim) {'2D' if with_2d_quantization else '1D'} max_abs_err={max_abs:.6f}")

    if max_abs > 1e-4:
        print("\n[FAIL Debug] Scale Mismatch (Top-Left 4x4 Unique Blocks):")
        print(f"SUT (Unique):\n{scales_sut[:4,:4]}")
        print(f"REF (Unique):\n{scales_ref[:4,:4]}")
    torch.testing.assert_close(
        x_hat_sut, x_hat_ref, atol=1e-4, rtol=0,
        msg=f"Rowwise mismatch M={M} N={N}"
    )

    # 7. Transpose (Colwise) Check
    # ----------------------------------------------------
    if return_transpose and x_mxfp4_sut._columnwise_data is not None:
        
        qx_t_raw = x_mxfp4_sut._columnwise_data.view(dtype=torch.uint8)
        sx_t_raw = x_mxfp4_sut._columnwise_scale_inv
        
        # Logical Transpose Shape: (N, M)
        qx_t_sut = qx_t_raw[:N, :M].contiguous()
        
        scale_cols_t = math.ceil(M / 32)
        
        # Transpose logic is 1D (per row of the transposed matrix, which is a col of original)
        # NOTE: MXFP4 Colwise is almost always 1D blocked (1x32 along K dimension).
        # It is rarely 2D blocked in the same way.
        # Assuming 1D for Colwise (standard GEMM requirement):
        sx_t_sut_unique = sx_t_raw[:N, :scale_cols_t].contiguous()
        
        scales_t_sut = e8m0_to_scale(sx_t_sut_unique)
        s_t_exp = scales_t_sut.repeat_interleave(32, dim=1)
        scales_t_sut_expanded = s_t_exp[:N, :M]

        if SIMULATE_MXFP4_WITH_FP8:
            p_vals_t_sut = fp8_e4m3_to_float(qx_t_sut)
        else:
            p_vals_t_sut, _ = unpack_fp4_to_float(qx_t_sut)

        x_hat_t_sut = (p_vals_t_sut * scales_t_sut_expanded).to(torch.float32)

        # Ref Colwise
        qx_t_ref_vals, _ = unpack_fp4_to_float(x_mxfp4_ref.data_t.view(dtype=torch.uint8))
        scales_t_ref = e8m0_to_scale(x_mxfp4_ref.scale_t)
        
        # Ref usually returns correct shape, assume 1D expansion for transpose
        s_t_ref_exp = scales_t_ref.repeat_interleave(32, dim=1)
        scales_t_ref_expanded = s_t_ref_exp[:N, :M]
        
        x_hat_t_ref = (qx_t_ref_vals * scales_t_ref_expanded).to(torch.float32)

        torch.testing.assert_close(
            x_hat_t_sut, x_hat_t_ref, atol=1e-4, rtol=0,
            msg=f"Colwise mismatch M={M} N={N}"
        )




@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (256, 256),
        (256, 1024),
        (1024, 256),
        # # Padding required cases
        (256, 272*2),
        (304, 304),
        (320, 256),
        # # Some larger tiles
        (2048, 2048),
        (1024, 2048),
        (2048, 1024),
        # # # largest tile
        (8192, 8192),   # Standard Tile
    ],
)


@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "return_transpose", [True, False], ids=["quantize_transpose", "skip_transpose"]
)
@pytest.mark.parametrize(
    "use_cpp_allocator", [True,False], ids=["cpp_allocator","python_allocator"]
)
@pytest.mark.parametrize(
    "global_scaling", [False,True], ids=str
)
@pytest.mark.parametrize(
    "with_2d_quantization", [ False,True], ids=["1d_quantization","2d_quantization"]
)
@pytest.mark.parametrize(
    "encode", [ False,True], ids=["decode_centric","encode_centric"]
)
def test_mxfp4_block_correctness(
    x_dtype, M, N, return_transpose, use_cpp_allocator,global_scaling, with_2d_quantization,encode
):
    check_quantization_mxfp4_versus_reference(
        x_dtype, M, N, return_transpose, use_cpp_allocator,global_scaling, with_2d_quantization,encode
    )

@pytest.mark.parametrize("M, N", [(128, 128)])
def test_mxfp4_extrema(M, N):
    """Check handling of zeros and max values."""
    device = "cuda"
    
    # Case 1: Zeros
    # Log2(0) is -inf, but we clamp. Scale should be minimal.
    x_zero = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    ref = MXFP4QuantizerRef().quantize(x_zero)
    # Scale should be -127 biased -> 0
    assert torch.all(ref.scale == 0), "Zero input did not produce zero scale index"
    
    # Case 2: Large values
    # Max valid float should not crash it
    x_max = torch.full((M, N), 65000.0, dtype=torch.bfloat16, device=device)
    # Just ensure it runs without segfault
    q = MXFP4Quantizer(with_rht=False)
    q.quantize(x_max)

# =============================================================================
# 4. RHT Integration Test
# =============================================================================

def test_mxfp4_rht_integration():
    """
    Verifies that enabling RHT changes the data distribution before quantization.
    """
    device = "cuda"
    M, N = 128, 128
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    
    # Add a massive outlier
    x[0, 0] = 100000.0
    
    # 1. Quantize WITHOUT RHT
    q_no_rht = MXFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=False,
    ).quantize(x)

    q_rht = MXFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
    ).quantize(x)

    scale_no_rht = q_no_rht._columnwise_scale_inv[0, 0].item()
    scale_rht = q_rht._columnwise_scale_inv[0, 0].item()
    
    print(f"Scale No-RHT: {scale_no_rht}")
    print(f"Scale RHT:    {scale_rht}")
    print(torch.sum(q_rht._columnwise_scale_inv-q_no_rht._columnwise_scale_inv))
    
    # With RHT, the outlier 1000.0 should be smeared.
    # 1000 / 4 = 250. 
    # log2(1000) approx 10. log2(250) approx 8.
    # The scale exponent should be SMALLER with RHT.
    
    assert scale_rht < scale_no_rht, "RHT failed to suppress outlier!"