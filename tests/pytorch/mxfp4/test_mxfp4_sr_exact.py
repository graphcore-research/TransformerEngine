# test_mxfp4_sr.py
# Copyright (c) 2025, Graphcore / Generic AI.
# Adapted from NVIDIA NVFP4 SR tests.

import pytest
import torch
import math
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP4Quantizer

# Config
SIMULATE_MXFP4_WITH_FP8 = True

# =============================================================================
# 1. Decoding Helpers (Dequantizer)
# =============================================================================

def e8m0_to_scale(sx: torch.Tensor) -> torch.Tensor:
    if sx.is_floating_point(): return sx.to(torch.float32)
    exp = sx.to(torch.int16) - 127
    return torch.pow(torch.tensor(2.0, dtype=torch.float32, device=sx.device), exp.to(torch.float32))

def fp8_e4m3_to_float(qx_bytes: torch.Tensor) -> torch.Tensor:
    # See previous file for full implementation details
    x = qx_bytes.to(torch.int16)
    sign = torch.where(((x >> 7) & 1) == 0, 1.0, -1.0).to(torch.float32)
    exp = ((x >> 3) & 0xF).to(torch.int16)
    mant = (x & 0x7).to(torch.float32)
    bias = 7
    
    out = torch.zeros_like(mant, device=qx_bytes.device)
    # Normal
    is_normal = (exp != 0) & (exp != 0xF)
    if is_normal.any():
        out[is_normal] = sign[is_normal] * torch.pow(2.0, (exp[is_normal]-bias).float()) * (1.0 + mant[is_normal]/8.0)
    # Subnorm
    is_subnorm = (exp == 0) & (mant != 0)
    if is_subnorm.any():
        out[is_subnorm] = sign[is_subnorm] * (2.0**(1-bias)) * (mant[is_subnorm]/8.0)
    return out

def dequantize_mxfp4_rowwise(q_obj, N_cols):
    """Reconstructs float tensor from MXFP4Quantized object."""
    qx = q_obj._rowwise_data.view(dtype=torch.uint8)
    sx = q_obj._rowwise_scale_inv
    
    # Handle padding
    blocks = N_cols // 32
    sx = sx[:, :blocks]
    
    # Decode
    scales = e8m0_to_scale(sx).repeat_interleave(32, dim=1)
    
    if SIMULATE_MXFP4_WITH_FP8:
        vals = fp8_e4m3_to_float(qx)
    else:
        # Fallback for real FP4 if needed (not implemented here)
        raise NotImplementedError("SR Test currently assumes FP8 simulation path for decoding")
        
    return (vals * scales) / 6.0

# =============================================================================
# 2. Test Function
# =============================================================================

@pytest.mark.parametrize("M, N", [(1024, 1024)])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_mxfp4_stochastic_rounding(M, N, x_dtype):
    """
    Checks if Stochastic Rounding (SR) provides lower error on average 
    compared to Round-to-Nearest (RN) for inputs that lie between quantization steps.
    """
    device = "cuda"
    torch.manual_seed(12345)
    
    # 1. Create Input
    # We use a range that hits many "0.5" intervals where RN has max error
    # and SR shines.
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    
    # 2. Setup Quantizer Dtype
    if SIMULATE_MXFP4_WITH_FP8:
        alloc_dtype = tex.DType.kFloat8E4M3
    else:
        alloc_dtype = tex.DType.kFloat4E2M1

    # 3. Baseline: Round to Nearest (SR=False)
    # ----------------------------------------
    q_rn = MXFP4Quantizer(
        fp4_dtype=alloc_dtype,
        rowwise=True,
        stochastic_rounding=False  # <--- OFF
    ).quantize(x)
    
    dq_rn = dequantize_mxfp4_rowwise(q_rn, N)
    
    # Calculate RMSE for RN
    error_rn = (dq_rn - x).float()
    rmse_rn = torch.sqrt((error_rn**2).mean()).item()
    
    print(f"\nMXFP4 Stochastic Rounding Test (M={M}, N={N})")
    print(f"RMSE (Round-Nearest):     {rmse_rn:.6e}")

    # 4. Test: Stochastic Rounding (SR=True)
    # --------------------------------------
    # We run this multiple times and average the DEQUANTIZED results.
    # The average of SR should approach the true value x better than RN.
    
    n_iters = 50
    sum_dq_sr = torch.zeros_like(x, dtype=torch.float32)
    
    quantizer_sr = MXFP4Quantizer(
        fp4_dtype=alloc_dtype,
        rowwise=True,
        stochastic_rounding=True   # <--- ON
    )
    
    for i in range(n_iters):
        # Note: We must create new random seeds or ensure the kernel
        # actually draws new random numbers every call. 
        # (TE kernels usually use internal philox counters or global seed).
        q_sr = quantizer_sr.quantize(x)
        dq_sr = dequantize_mxfp4_rowwise(q_sr, N)
        sum_dq_sr += dq_sr.float()

    avg_dq_sr = sum_dq_sr / n_iters
    
    # Calculate RMSE for SR Average
    error_sr = (avg_dq_sr - x).float()
    rmse_sr = torch.sqrt((error_sr**2).mean()).item()

    print(f"RMSE (Stochastic Avg 50): {rmse_sr:.6e}")
    
    # 5. Assertion
    # --------------------------------------
    # SR Average error should be strictly less than RN error
    assert rmse_sr < rmse_rn, (
        f"Stochastic Rounding failed to improve RMSE! "
        f"RN={rmse_rn:.6e}, SR={rmse_sr:.6e}"
    )
    
    improvement = (rmse_rn - rmse_sr) / rmse_rn * 100
    print(f"PASS: SR improved RMSE by {improvement:.2f}%")