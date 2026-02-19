# test_mxfp4_rht_exact.py
# Copyright (c) 2025, Graphcore / Generic AI.

import pytest
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import MXFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils

# =============================================================================
# CONFIGURATION
# =============================================================================
SIMULATE_MXFP4_WITH_FP8 = True 

# =============================================================================
# HELPERS
# =============================================================================

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

def generate_cuda_compatible_hadamard(dim: int,
                                      random_sign_mask: int = 0,
                                      inverse: bool = False,
                                      device="cuda") -> torch.Tensor:
    assert dim == 16, "Kernel hardcodes dim=16"

    r = torch.arange(dim, dtype=torch.int32, device=device).view(dim, 1)
    c = torch.arange(dim, dtype=torch.int32, device=device).view(1, dim)

    dot = r & c
    popc = (dot & 1) + ((dot >> 1) & 1) + ((dot >> 2) & 1) + ((dot >> 3) & 1)
    parity = popc & 1  # only LSB matters (matches sign_i<<31 behavior)

    # MATCH CUDA:
    # non-inverse: mask uses c
    # inverse:     mask uses r
    if inverse:
        mask_bit = (random_sign_mask >> r) & 1
    else:
        mask_bit = (random_sign_mask >> c) & 1

    sign_bit = mask_bit ^ parity
    mat = (1.0 - 2.0 * sign_bit.float()) * 0.25  # k16x16HadamardScale

    return mat.to(torch.bfloat16)


def apply_hadamard_block16_exact(x: torch.Tensor, h_matrix: torch.Tensor) -> torch.Tensor:
    """Apply H_16 to blocks of input x."""
    M, N = x.shape
    H_dim = 16
    assert N % H_dim == 0
    
    # [M, N/16, 16] @ [16, 16] -> [M, N/16, 16]
    x_view = x.view(M, N // H_dim, H_dim)
    
    # Note on Transpose:
    # RHT usually defines Y = X @ H. 
    # Since H is symmetric, H == H.T.
    # We use standard matmul.
    x_out = torch.matmul(x_view, h_matrix)
    
    return x_out.view(M, N)

# =============================================================================
# DECODING HELPERS
# =============================================================================
def unpack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated

def decode_e2m1_to_float(idx: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], 
                       dtype=torch.float32, device=idx.device)
    return lut[idx.long()]

def decode_e4m3_to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.view(torch.float8_e4m3fn).to(torch.float32)

def e8m0_to_scale(sx: torch.Tensor) -> torch.Tensor:
    if sx.is_floating_point(): return sx.float()
    exp = sx.to(torch.int32) - 127
    return torch.pow(2.0, exp.float())

# =============================================================================
# TEST IMPLEMENTATION
# =============================================================================
def check_quantization_mxfp4_versus_reference(
    x_dtype: torch.dtype, M: int, N: int, contiguous: bool, 
    return_transpose: bool, use_cpp_allocator: bool, 
    with_rht: bool = True, with_post_rht_amax: bool = True, global_scaling: bool = False
) -> None:
    
    te_dtype = tex.DType.kFloat8E4M3 if SIMULATE_MXFP4_WITH_FP8 else tex.DType.kFloat4E2M1
    device = "cuda"
    torch.manual_seed(42)
    
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    if not contiguous: x = x.transpose(0, 1).contiguous().transpose(0, 1)

    # 1. Run SUT
    mxfp4_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_rht=with_rht,
        with_post_rht_amax=with_post_rht_amax,
        global_scaling=global_scaling
    )

    if use_cpp_allocator:
        x_sut = mxfp4_quantizer(x)
    else:
        x_sut = mxfp4_quantizer.make_empty(x.shape, dtype=x_dtype, device=device)
        x_sut = mxfp4_quantizer.update_quantized(x, x_sut)

    # 2. Extract SUT Data
    qx_sut_raw = x_sut._rowwise_data.view(dtype=torch.uint8)
    sx_sut_raw = x_sut._rowwise_scale_inv
    sx_sut_float = e8m0_to_scale(sx_sut_raw)
    
    if SIMULATE_MXFP4_WITH_FP8:
        val_sut_float = decode_e4m3_to_float(qx_sut_raw)
    else:
        val_sut_float = decode_e2m1_to_float(unpack_fp4_to_uint8(qx_sut_raw))

    sx_sut_exp = sx_sut_float.repeat_interleave(32, dim=1)
    if sx_sut_exp.shape[1] > N: sx_sut_exp = sx_sut_exp[:, :N]
    x_rec_sut = val_sut_float * sx_sut_exp

    # 3. Reference
    x_ref_input = x.clone()
    
    ref_quantizer = MXFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        quant_tile_shape=(1, 32),
        with_rht=True, 
        pow_2_scales=True,
        with_random_sign_mask=True,
        use_global_scale=global_scaling
    )
    x_ref = ref_quantizer.quantize(x_ref_input)
    
    qx_ref_raw = x_ref.data.view(dtype=torch.uint8)
    sx_ref_raw = x_ref.scale.view(dtype=torch.uint8)
    
    val_ref_float = decode_e2m1_to_float(unpack_fp4_to_uint8(qx_ref_raw))
    sx_ref_float = e8m0_to_scale(sx_ref_raw)
    sx_ref_exp = sx_ref_float.repeat_interleave(32, dim=1)[:, :N]
    x_rec_ref = val_ref_float * sx_ref_exp

    # 4. Compare Scales (The real verification)
    print(f"\n[{M}x{N}] Compare Scales (RHT={with_rht})")
    rows, cols = sx_ref_float.shape
    sx_sut_val = sx_sut_float[:rows, :cols]
    
    # Allow +/- 1 exponent diff (factor of 2) rarely due to float accumulation differences in AMAX
    # But generally should match exactly.
    diff = (sx_sut_val - sx_ref_float).abs()
    mismatch_mask = diff > 1e-5
    
    if mismatch_mask.any():
        print(f"FAILED: {mismatch_mask.sum()} scale mismatches")
        print("SUT Sample:", sx_sut_val[mismatch_mask][:5])
        print("Ref Sample:", sx_ref_float[mismatch_mask][:5])
        
        # If the difference is exactly factor of 4 (exponent 2), the 0.25 scale is still missing somewhere.
        ratio = sx_sut_val[mismatch_mask] / sx_ref_float[mismatch_mask]
        print("Ratio Sample:", ratio[:5])
        
        pytest.fail("Rowwise Scale Mismatch")
    else:
        print(">> PASS: Scales Match")

@pytest.mark.parametrize("M, N", [(128, 128), (256, 1024), (1024, 2048)])
@pytest.mark.parametrize("return_transpose", [True, False])
@pytest.mark.parametrize(
    "global_scaling", [False,True], ids=str
)
def test_mxfp4_rht_quantization(M, N, return_transpose,global_scaling):
    check_quantization_mxfp4_versus_reference(
        x_dtype=torch.bfloat16, M=M, N=N, contiguous=True, 
        return_transpose=return_transpose, use_cpp_allocator=True, with_rht=True, global_scaling=global_scaling
    )