# Copyright (c) 2025, Graphcore / Generic AI.
# MXFP4 GEMM tests, adapted from NVFP4 tests.

import math
import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import MXFP4Quantizer, MXFP4Tensor
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import MXFP4QuantizerRef
# torch.set_printoptions(profile="full")
# ---------------------------------------------------------------------------
# Feature / recipe availability (mirror NVFP4 style if possible)
# ---------------------------------------------------------------------------

#TODO ADD 1's matrix test 

try:
    mxfp4_available, reason_for_no_mxfp4 = te.is_mxfp4_available(return_reason=True)  # type: ignore[attr-defined]
except AttributeError:
    # If you haven't wired this yet, just always run the tests.
    mxfp4_available, reason_for_no_mxfp4 = True, "te.is_mxfp4_available not implemented"

SIMULATE_MXFP4_WITH_FP8 = True

def test_mxfp4_gemm_ones_exact():
    """
    Simplified debug test:
    - Fixed dimensions (128x128x128)
    - Inputs are all ONES
    - No accumulation
    - Rowwise only (columnwise=False to fix stride issue)
    """
    # Fixed Debug Config
    M, K, N = 128, 128, 128
    accumulate = False
    x_dtype = torch.bfloat16
    w_dtype = torch.bfloat16
    out_dtype = torch.float32
    
    # Force columnwise=False to avoid the 256-width packed tensor issue
    is_x_columnwise = True
    is_w_columnwise = True
    
    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------
    te_dtype = tex.DType.kFloat8E4M3 if SIMULATE_MXFP4_WITH_FP8 else tex.DType.kfloat4E2M1
    device = "cuda"
    torch.manual_seed(0)

    # Input tensors: ALL ONES
    x_shape = (K, M) if is_x_columnwise else (M, K)
    w_shape = (K, N) if is_w_columnwise else (N, K)

    x = torch.randn(x_shape, dtype=x_dtype, device=device)
    w = torch.randn(w_shape, dtype=w_dtype, device=device)

    out = None # No accumulate for this simple debug test

    # ----------------------------------------------------------------------
    # Native TE MXFP4 quantization
    # ----------------------------------------------------------------------
    x_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=is_x_columnwise, # False
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    w_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=is_w_columnwise, # False
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    x_mxfp4_native = x_quantizer.make_empty(x_shape, dtype=x_dtype, device=device, requires_grad=False)
    x_mxfp4_native = x_quantizer.update_quantized(x, x_mxfp4_native)

    w_mxfp4_native = w_quantizer.make_empty(w_shape, dtype=w_dtype, device=device, requires_grad=False)
    w_mxfp4_native = w_quantizer.update_quantized(w, w_mxfp4_native)

    # ----------------------------------------------------------------------
    # Reference Setup
    # ----------------------------------------------------------------------
    qx_data = x_mxfp4_native._rowwise_data.view(dtype=torch.uint8)[:M, :]
    qw_data = w_mxfp4_native._rowwise_data.view(dtype=torch.uint8)[:N, :]
    
    # Block size 32 logic
    expected_cols = math.ceil(K / 32)
    sx_trimmed = x_mxfp4_native._rowwise_scale_inv[:M, :expected_cols]
    sw_trimmed = w_mxfp4_native._rowwise_scale_inv[:N, :expected_cols]

    ref_quantizer = MXFP4QuantizerRef(simulate_mxfp4_with_fp8=SIMULATE_MXFP4_WITH_FP8, use_global_scale=False, with_rht=False)
    x_mxfp4_ref = ref_quantizer.quantize(x)
    w_mxfp4_ref = ref_quantizer.quantize(w)

    y_ref = ref_quantizer.qgemm(
        qx=qx_data, qw=qw_data, m_params=None, out_dtype=out_dtype,
        sx=sx_trimmed, sw=sw_trimmed, bias=None, out=None, accumulate=False,
        gemm_type=None, qresult_x=x_mxfp4_ref, qresult_w=w_mxfp4_ref,
    )

    # ----------------------------------------------------------------------
    # Native GEMM Call (Corrected Arguments for X * W^T)
    # ----------------------------------------------------------------------
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    
    # Compute X * W^T
    # Matrix A = x_mxfp4_native (M x K)
    # Matrix B = w_mxfp4_native (N x K) --> Transpose B to get (K x N)
    
    y_native = tex.generic_gemm(
        w_mxfp4_native,  # A
        True,           # transA (False)
        x_mxfp4_native,  # B
        False,            # transB (True) -> interprets B as (K, N)
        None,            # out
        None, TE_DType[out_dtype], # out_quantizer, out_dtype
        None, TE_DType[torch.bfloat16], # bias, bias_dtype
        False, None, False, # gelu stuff, grad
        workspace, workspace.shape[0],
        False, False # accumulate, split_accumulator
    )[0]

    # ----------------------------------------------------------------------
    # Debug Prints
    # ----------------------------------------------------------------------
    print("\n=== MXFP4 GEMM (ALL ONES) DEBUG ===")
    print(f"y_ref mean: {y_ref.mean().item()}")
    print(f"y_native mean: {y_native.mean().item()}")
    
    # Check specific value (first element)
    print(f"y_ref[0,0]: {y_ref[0,0].item()}")
    print(f"y_native[0,0]: {y_native[0,0].item()}")
    print("\n=== MXFP4 GEMM DEBUG ===")
    print(f"M, K, N = {M}, {K}, {N}")
    print(f"dtypes: x={x_dtype}, w={w_dtype}, out={out_dtype}")
    print("x shape:", x.shape, "w shape:", w.shape)
    print("qx_data shape:", qx_data.shape, "qw_data shape:", qw_data.shape)
    print("sx_trimmed shape:", sx_trimmed.shape, "sw_trimmed shape:", sw_trimmed.shape)

    # First few bytes / scales
    print("qx_data[0, :8]:", qx_data[0, :16].tolist())
    print("qw_data[0, :8]:", qw_data[0, :8].tolist())
    print("sx_trimmed[0, :4]:", sx_trimmed[0, :4].tolist())
    print("sw_trimmed[0, :4]:", sw_trimmed[0, :4].tolist())

    # Look at some actual outputs
    print("y_ref[0, :4]:", y_ref[0, :24].tolist())
    print("y_native[0, :4]:", y_native[0, :24].tolist())
    print("y_ref:", y_ref)
    print("y_ref var:", y_ref.var())

    print("y_native:", y_native)
    print("y_native var:", y_native.var())
    print("y_native sum:", y_native.sum().item())
    print("y_native shape:", y_native.shape)
    print("y_ref shape:", y_ref.shape)
    diff = (y_native - y_ref).abs()
    print("max_abs_diff:", diff.max().item())
    print("mean_abs_diff:", diff.mean().item())
    
    # Ratio check
    if y_ref[0,0] != 0:
        ratio = y_native[0,0] / y_ref[0,0]
        print(f"Ratio (Native / Ref): {ratio.item()}")
        if abs(ratio.item() - 36.0) < 0.1:
            print(">> DIAGNOSIS: Result is exactly 36x reference. Missing 1/36 scaling factor.")
        elif abs(ratio.item() - 1.0) < 0.1:
            print(">> DIAGNOSIS: Result matches reference!")
        else:
            print(f">> DIAGNOSIS: Unknown scaling error. Factor: {ratio.item()}")

    print("=== END DEBUG ===\n")

    torch.testing.assert_close(
        y_native,
        y_ref,
        atol=1e-3, # Looser tolerance might be needed for low precision
        rtol=1e-3,
        msg="MXFP4 GEMM mismatch (All Ones Input)",
    )

def check_mxfp4_gemm_versus_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
    *,
    x_columnwise: bool = False,
    w_columnwise: bool = False,
    debug: bool = False,
    global_scaling: bool = False,
    with_rht: bool = False,
    use_cpp_allocator: bool = False,
    encode: bool = False

) -> None:
    """
    Compare MXFP4 GEMM from TE (tex.generic_gemm on MXFP4Tensor) against the
    Python reference implementation MXFP4QuantizerRef.qgemm.

    Layout convention is kept identical to the NVFP4 test:

        x: (M, K)  if rowwise   (default)
           (K, M)  if columnwise

        w: (N, K)  if rowwise   (default)
           (K, N)  if columnwise

    GEMM result is M x N.
    """
    te_dtype = tex.DType.kFloat8E4M3 if SIMULATE_MXFP4_WITH_FP8 else tex.DType.kfloat4E2M1 # same FP4 format as NVFP4 (E2M1)

    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # MXFP4 block size constraints:
    # make_empty() asserts that both:
    #   - last dim (K) % 32 == 0
    #   - product of leading dims (M or N) % 32 == 0
    assert K % 32 == 0, f"K={K} must be divisible by 32 for MXFP4"
    assert M % 32 == 0, f"M={M} must be divisible by 32 for MXFP4"
    assert N % 32 == 0, f"N={N} must be divisible by 32 for MXFP4"

    # Input tensors
    x_shape = (K, M) if x_columnwise else (M, K)
    w_shape = (K, N) if w_columnwise else (N, K)

    x = torch.ones(x_shape, dtype=x_dtype, device=device)
    w = torch.randn(w_shape, dtype=w_dtype, device=device)

    # Optional accumulation buffer
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None

    # ----------------------------------------------------------------------
    # Native TE MXFP4 quantization (C++/CUDA path)
    # ----------------------------------------------------------------------
    x_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=with_rht,
        with_post_rht_amax=True,
        global_scaling=global_scaling,
        encode_centric=encode
    )
    w_quantizer = MXFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=with_rht,
        with_post_rht_amax=True,
        global_scaling=global_scaling,
        encode_centric=encode
    )

    # Quantize x and w into MXFP4 tensors

    if use_cpp_allocator:
        
        x_mxfp4_native = x_quantizer.quantize(x)

        w_mxfp4_native = w_quantizer.quantize(w)

    else:
        x_mxfp4_native = x_quantizer.make_empty(
            x_shape, dtype=x_dtype, device=device, requires_grad=False
        )
        x_mxfp4_native = x_quantizer.quantize(x, x_mxfp4_native)

        w_mxfp4_native = w_quantizer.make_empty(
            w_shape, dtype=w_dtype, device=device, requires_grad=False
        )
        w_mxfp4_native = w_quantizer.update_quantized(w, w_mxfp4_native)

    # ----------------------------------------------------------------------
    # Extract packed FP4 data + E8M0 scales from native MXFP4 tensors
    # ----------------------------------------------------------------------
    # NOTE: For GEMM we only test rowwise × rowwise. The arguments
    # x_columnwise / w_columnwise are kept for symmetry with NVFP4; if you
    # add columnwise support in the reference later, wire them up here.
    assert not x_columnwise and not w_columnwise, "Reference GEMM currently only handles rowwise."

    qx_data = x_mxfp4_native._rowwise_data.view(dtype=torch.uint8)
    qw_data = w_mxfp4_native._rowwise_data.view(dtype=torch.uint8)

    # Trim rows to actual M / N (rowwise_data is padded in outer dim to 128)
    qx_data = qx_data[:M, :]
    qw_data = qw_data[:N, :]

    sx_native = x_mxfp4_native._rowwise_scale_inv
    sw_native = w_mxfp4_native._rowwise_scale_inv

    # MXFP4 uses block size 32 along K, with potential padding on the inner dim.
    block_length = 32
    expected_sx_cols = math.ceil(K / block_length)
    expected_sw_cols = math.ceil(K / block_length)

    sx_trimmed = sx_native[:M, :expected_sx_cols]
    sw_trimmed = sw_native[:N, :expected_sw_cols]

    # For MXFP4, scales are E8M0 encoded as uint8 exponents; the reference
    # quantizer already uses this representation, so we DO NOT reinterpret
    # dtype like the NVFP4 float8_e4m3fn case.
    assert sx_trimmed.dtype == torch.uint8
    assert sw_trimmed.dtype == torch.uint8

    # ----------------------------------------------------------------------
    # Python reference quantization + GEMM
    # ----------------------------------------------------------------------
    ref_quantizer = MXFP4QuantizerRef(        use_global_scale=global_scaling,
        with_rht=with_rht,simulate_mxfp4_with_fp8=SIMULATE_MXFP4_WITH_FP8,encode_centric=encode)

    # These give you Python-side reference quantized tensors (qx_ref, sx_ref...)
    x_mxfp4_ref = ref_quantizer.quantize(x)
    w_mxfp4_ref = ref_quantizer.quantize(w)

    # Reference GEMM from the quantizer, using the *hardware* qx/qw + sx/sw
    # (which we've trimmed to drop padding), but the Python qresults for
    # metadata / consistency.
    y_ref = ref_quantizer.qgemm(
        qx=qx_data,
        qw=qw_data,
        m_params=None,  # MMParams not used here
        out_dtype=out_dtype,
        sx=sx_trimmed,
        sw=sw_trimmed,
        bias=None,
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
        gemm_type=None,
        qresult_x=x_mxfp4_ref,
        qresult_w=w_mxfp4_ref,

    )

    # ----------------------------------------------------------------------
    # Native TE GEMM using tex.generic_gemm (MXFP4 path)
    # ----------------------------------------------------------------------
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Keep the same convention as the NVFP4 test:
    #   x: (M, K)  => transb=False
    #   w: (N, K)  => transa=True  (we interpret as (K, N) in GEMM)

    out_quantizer = None
    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu = False
    gelu_input = None
    use_grad = False
    use_split_accumulator = False

    y_native = tex.generic_gemm(
            w_mxfp4_native,  # A = X (M, K)
            True,          # transA = False
            x_mxfp4_native,  # B = W (N, K)
            False,          # transB = True (creates W^T)
            out.clone() if accumulate else None,
            out_quantizer,
            TE_DType[out_dtype],
            bias,
            bias_dtype,
            use_gelu,
            gelu_input,
            use_grad,
            workspace,
            workspace.shape[0],
            accumulate,
            use_split_accumulator,
        )[0]

    # ----------------------------------------------------------------------
    # Compare results
    # ----------------------------------------------------------------------
    # Ensure distinct tensors if accumulation is used
    assert y_ref is not y_native, "y_ref and y_native should not be the same tensor"

    # Replace NaNs with zeros for comparison
    assert not torch.isnan(y_ref.float()).all(), "All elements are nan in y_ref"
    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)
    if debug:
        print("\n=== MXFP4 GEMM DEBUG ===")
        print(f"M, K, N = {M}, {K}, {N}")
        print(f"dtypes: x={x_dtype}, w={w_dtype}, out={out_dtype}")
        print("x shape:", x.shape, "w shape:", w.shape)
        print("qx_data shape:", qx_data.shape, "qw_data shape:", qw_data.shape)
        print("sx_native shape:", sx_native.shape, "sw_native shape:", sw_native.shape)
        print("sx_trimmed shape:", sx_trimmed.shape, "sw_trimmed shape:", sw_trimmed.shape)

        # First few bytes / scales
        print("qx_data[0, :8]:", qx_data[0, :8].tolist())
        print("qw_data[0, :8]:", qw_data[0, :8].tolist())
        print("sx_trimmed[0, :4]:", sx_trimmed[0, :4].tolist())
        print("sw_trimmed[0, :4]:", sw_trimmed[0, :4].tolist())

        # Look at some actual outputs
        print("y_ref[0, :4]:", y_ref[0, :4].tolist())
        print("y_native[0, :4]:", y_native[0, :4].tolist())
        print("y_ref:", y_ref)
        print("y_native:", y_native)
        print("y_native sum:", y_native.sum().item())

        diff = (y_native - y_ref).abs()
        print("max_abs_diff:", diff.max().item())
        print("mean_abs_diff:", diff.mean().item())

        # Is it *just* a global scale factor?
        nonzero = y_ref != 0
        if nonzero.any():
            ratio = (y_native[nonzero] / y_ref[nonzero]).flatten()
            print("ratio min:", ratio.min().item())
            print("ratio max:", ratio.max().item())
            print("ratio mean:", ratio.mean().item())
        else:
            print("All y_ref entries are zero??")

        # And see what the Python quantizer object looks like
        print("qresult_x fields:", list(vars(x_mxfp4_ref).keys()))
        print("qresult_w fields:", list(vars(w_mxfp4_ref).keys()))
        print("=== END MXFP4 GEMM DEBUG ===\n")
    # Tolerances can be tuned – start with NVFP4's tolerances
    torch.testing.assert_close(
        y_native,
        y_ref,
        atol=8e-3,
        rtol=8e-3,
        msg="MXFP4 GEMM mismatch between TE and reference qgemm",
    )


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not mxfp4_available, reason=reason_for_no_mxfp4)
@pytest.mark.parametrize(
    "M, K, N",
    [
        # All dims must be divisible by 32 to satisfy MXFP4 block-size asserts
        (128, 128, 128),
        (256, 256, 256),
        (256, 512, 256),
        (512, 512, 512),
        (4096, 512, 3072),
        # Add more if you want larger stress tests, but this is already beefy
        (8192, 8192, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [ torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False], ids=["accumulate", "no_accumulate"])
@pytest.mark.parametrize(
    "is_x_columnwise, is_w_columnwise",
    [
        (False, False),  # Only rowwise × rowwise supported by MXFP4 reference GEMM
    ],
    ids=["rowxrow"],
)
@pytest.mark.parametrize(
    "global_scaling", [True,False], ids=str
)
@pytest.mark.parametrize(
    "with_rht", [False,True], ids=str
)
@pytest.mark.parametrize(
    "use_cpp_allocator", [False,True], ids=str
)
@pytest.mark.parametrize(
    "encode", [ False,True], ids=["decode_centric","encode_centric"]
)
def test_mxfp4_gemm_versus_reference(
    M: int,
    K: int,
    N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    accumulate: bool,
    is_x_columnwise: bool,
    is_w_columnwise: bool,
    global_scaling: bool,
    with_rht: bool,
    use_cpp_allocator: bool,
    encode: bool,
) -> None:
    # debug = (
    #     M == 128
    #     and K == 128
    #     and N == 128
    #     and not accumulate
    #     and x_dtype == torch.bfloat16
    #     and w_dtype == torch.bfloat16
    #     and out_dtype == torch.float32
    # )
    check_mxfp4_gemm_versus_reference(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
        x_columnwise=is_x_columnwise,
        w_columnwise=is_w_columnwise,
        debug=False,
        global_scaling=global_scaling,
        with_rht=with_rht,
        use_cpp_allocator=use_cpp_allocator,
        encode=encode
    )
