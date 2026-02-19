# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.custom_recipes import quantization_mxfp4
from transformer_engine.pytorch.custom_recipes import utils

# ---------------------------------------------------------------------------
# Feature Check
# ---------------------------------------------------------------------------

try:
    mxfp4_available, reason_for_no_mxfp4 = te.is_mxfp4_available(return_reason=True)
except AttributeError:
    # Fallback for dev environments
    mxfp4_available, reason_for_no_mxfp4 = True, "te.is_mxfp4_available not implemented"

# ---------------------------------------------------------------------------
# Recipe Construction
# ---------------------------------------------------------------------------

class GetRecipes:
    @staticmethod
    def mxfp4_base(with_rht: bool = False, global_scaling: bool = False, encode: bool = False):
        """
        Constructs an MXFP4 recipe.
        
        Note: We assume recipe.MXFP4BlockScaling follows the same QParams 
        structure as NVFP4BlockScaling.
        """
        # 1. Instantiate the base recipe
        # If MXFP4BlockScaling isn't exposed yet, one might use DelayedScaling 
        # with specific args, but we assume the class exists for this rigorous test.
        try:
            mxfp4_recipe = recipe.MXFP4BlockScaling()
        except AttributeError:
            # Fallback if specific class missing, using generic with config
            mxfp4_recipe = recipe.DelayedScaling() 
        
        # 2. Configure QParams based on flags
        # MXFP4 constraints: Weights usually don't use RHT, Inputs/GradOutputs do.
        
        # Input Quantization
        mxfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(
            random_hadamard_transform=with_rht,
            global_scaling=global_scaling,
            encode_centric=encode
        )
        
        # Weight Quantization (Ignore 2D quant for now, usually no RHT for weights)
        mxfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(
            random_hadamard_transform=False, 
            global_scaling=global_scaling,
            fp4_2d_quantization=False,
            encode_centric=encode
        )
        
        # Grad Output Quantization
        mxfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(
            random_hadamard_transform=with_rht,
            global_scaling=global_scaling,
            encode_centric=encode
        )
        
        return mxfp4_recipe

# ---------------------------------------------------------------------------
# Reference Factory
# ---------------------------------------------------------------------------

def get_mxfp4_quantizer_factory(with_rht: bool = False, global_scaling: bool = False,encode: bool = False):
    """
    Create a quantizer factory for MXFP4 reference implementation.
    Args:
        with_rht: Enable Random Hadamard Transform on activations/grads
        global_scaling: Enable global scaling factor (1/36 vs dynamic)
    """

    def factory(role):
        # Common params for MXFP4 (E2M1, Block Size 32 implied by pow_2_scales=True)
        common_kwargs = {
            "dtype": utils.Fp4Formats.E2M1,
            "quant_tile_shape": (1, 32), # 1D Block scaling 32
            "pow_2_scales": True,
            "use_global_scale": global_scaling,
            "encode_centric": encode
        }

        if role == "linear_input":
            return quantization_mxfp4.MXFP4QuantizerRef(
                **common_kwargs,
                with_rht=with_rht,
                with_random_sign_mask=with_rht, # Usually implies random sign mask
            )
        elif role == "linear_weight":
            return quantization_mxfp4.MXFP4QuantizerRef(
                **common_kwargs,
                with_rht=False, # Weights usually static, no RHT
                with_random_sign_mask=False,
            )
        elif role == "linear_grad_output":
            return quantization_mxfp4.MXFP4QuantizerRef(
                **common_kwargs,
                with_rht=with_rht,
                with_random_sign_mask=with_rht,
            )
        elif role == "linear_output":
            # Output quantization usually not done or handled by next layer input
            return None
        elif role == "linear_grad_input":
            # Grad input quantization usually not done
            return None
        else:
            return None

    return factory


def reset_rng_states():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# ---------------------------------------------------------------------------
# Comparator Logic
# ---------------------------------------------------------------------------

def check_mxfp4_module_versus_reference(
    module_class,
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    with_rht: bool = False,
    global_scaling: bool = False,
    normalization: str = None, # For LayerNormLinear
    encode: bool = False
):
    """
    Compare native MXFP4 module against reference implementation.
    """
    device = "cuda"
    batch_size = 32
    seq_len = 128

    # MXFP4 Alignment Constraint
    assert in_features % 32 == 0, "in_features must be divisible by 32 for MXFP4"
    assert out_features % 32 == 0, "out_features must be divisible by 32 for MXFP4"

    # 1. Initialize Native Module
    reset_rng_states()
    print("\nCreate native module")
    
    kwargs = {
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias,
        "device": device,
        "params_dtype": x_dtype
    }
    
    if module_class == te.LayerNormLinear:
        kwargs["normalization"] = normalization
        kwargs["return_layernorm_output"] = True
        native_module = te.LayerNormLinear(**kwargs)
    else:
        native_module = te.Linear(**kwargs)

    # 2. Initialize Reference Module
    reset_rng_states()
    print("Create reference module")
    
    if module_class == te.LayerNormLinear:
        ref_module = te.LayerNormLinear(**kwargs)
    else:
        ref_module = te.Linear(**kwargs)

    # 3. Sync Weights
    with torch.no_grad():
        if hasattr(native_module, "weight") and hasattr(ref_module, "weight"):
            ref_module.weight.copy_(native_module.weight)
        if bias and hasattr(native_module, "bias") and hasattr(ref_module, "bias"):
            ref_module.bias.copy_(native_module.bias)
        
        # Sync LayerNorm params if applicable
        if hasattr(native_module, "layer_norm_weight"):
             if native_module.layer_norm_weight is not None:
                ref_module.layer_norm_weight.copy_(native_module.layer_norm_weight)
        if hasattr(native_module, "layer_norm_bias"):
             if native_module.layer_norm_bias is not None:
                ref_module.layer_norm_bias.copy_(native_module.layer_norm_bias)

    # 4. Create Recipes
    # Native
    mxfp4_recipe = GetRecipes.mxfp4_base(with_rht=with_rht, global_scaling=global_scaling,encode=encode)
    
    # Reference (injects Python QGEMM/Quantizer)
    mxfp4_ref_factory = get_mxfp4_quantizer_factory(with_rht=with_rht, global_scaling=global_scaling,encode=encode)
    mxfp4_ref_recipe = recipe.CustomRecipe(qfactory=mxfp4_ref_factory)

    # 5. Execution Loop
    native_outputs = []
    ref_outputs = []

    for step in range(num_steps):
        # Seed per step for data generation
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_shape = (batch_size, seq_len, in_features)
        x_val = torch.normal(mean=0.0, std=1.0, size=x_shape, dtype=x_dtype, device=device)
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_native.clone().detach().requires_grad_(True)

        grad_output_shape = (batch_size, seq_len, out_features)
        grad_output_val = torch.normal(
            mean=0.0, std=1.0, size=grad_output_shape, dtype=x_dtype, device=device
        )
        grad_output = grad_output_val.clone().detach()

        # --- Native Forward/Backward ---
        with te.autocast(enabled=True, recipe=mxfp4_recipe):
            if module_class == te.LayerNormLinear:
                y_native, ln_out_native = native_module(x_native, is_first_microbatch=(step == 0))
            else:
                y_native = native_module(x_native, is_first_microbatch=(step == 0))
                ln_out_native = None
        
        y_native.backward(grad_output)

        # --- Reference Forward/Backward ---
        with te.autocast(enabled=True, recipe=mxfp4_ref_recipe):
            if module_class == te.LayerNormLinear:
                y_ref, ln_out_ref = ref_module(x_ref)
            else:
                y_ref = ref_module(x_ref)
                ln_out_ref = None
                
        y_ref.backward(grad_output)

        # --- Store Results ---
        native_outputs.append({
            "output": y_native.detach().clone(),
            "ln_out": ln_out_native.detach().clone() if ln_out_native is not None else None,
            "input_grad": x_native.grad.detach().clone() if x_native.grad is not None else None,
            "weight_grad": native_module.weight.grad.detach().clone() if native_module.weight.grad is not None else None,
            "bias_grad": native_module.bias.grad.detach().clone() if bias and native_module.bias.grad is not None else None,
        })

        ref_outputs.append({
            "output": y_ref.detach().clone(),
            "ln_out": ln_out_ref.detach().clone() if ln_out_ref is not None else None,
            "input_grad": x_ref.grad.detach().clone() if x_ref.grad is not None else None,
            "weight_grad": ref_module.weight.grad.detach().clone() if ref_module.weight.grad is not None else None,
            "bias_grad": ref_module.bias.grad.detach().clone() if bias and ref_module.bias.grad is not None else None,
        })

    # 6. Comparison
    for step in range(num_steps):
        native_out = native_outputs[step]
        ref_out = ref_outputs[step]
        
        # Tolerance setup
        # MXFP4 is noisy, especially with RHT enabled. 
        # However, since we are comparing Bit-Exact logic (Python implementation of the algorithm)
        # vs CUDA kernel, we expect them to be relatively close if the math matches.
        # RHT adds matrix multiplications that might introduce small float errors.
        atol = 5e-3
        rtol = 5e-3

        # Compare Outputs
        torch.testing.assert_close(
            native_out["output"], ref_out["output"],
            atol=atol, rtol=rtol, msg=f"Output mismatch at step {step}"
        )
        
        # Compare LN Outputs (if applicable)
        if native_out["ln_out"] is not None:
             torch.testing.assert_close(
                native_out["ln_out"], ref_out["ln_out"],
                atol=atol, rtol=rtol, msg=f"LN Output mismatch at step {step}"
            )
        # Compare Input Gradients
        if native_out["input_grad"] is not None:
            torch.testing.assert_close(
                native_out["input_grad"], ref_out["input_grad"],
                atol=atol, rtol=rtol, msg=f"Input gradient mismatch at step {step}"
            )

        # Compare Weight Gradients
        # print(native_out["weight_grad"] - ref_out["weight_grad"] )
        if native_out["weight_grad"] is not None:
            torch.testing.assert_close(
                native_out["weight_grad"], ref_out["weight_grad"],
                atol=atol, rtol=rtol, msg=f"Weight gradient mismatch at step {step}"
            )

        # Compare Bias Gradients
        if bias and native_out["bias_grad"] is not None:
            torch.testing.assert_close(
                native_out["bias_grad"], ref_out["bias_grad"],
                atol=atol, rtol=rtol, msg=f"Bias gradient mismatch at step {step}"
            )


# ---------------------------------------------------------------------------
# Pytest Entry Points
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not mxfp4_available, reason=reason_for_no_mxfp4)
@pytest.mark.parametrize("in_features, out_features", [(128, 128), (256, 128)])
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
# RHT usually works best with BF16 inputs
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str) 
@pytest.mark.parametrize("num_steps", [1,3], ids=["single_step","multi_step"])
@pytest.mark.parametrize("with_rht", [True,False], ids=str)
@pytest.mark.parametrize("global_scaling", [False,True], ids=str)
@pytest.mark.parametrize("encode", [False, True], ids=["standard","encode_centric"])
def test_mxfp4_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int,
    with_rht: bool,
    global_scaling: bool,
    encode: bool
):
    """Test MXFP4 Linear module against reference implementation."""
    check_mxfp4_module_versus_reference(
        module_class=te.Linear,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        x_dtype=x_dtype,
        num_steps=num_steps,
        with_rht=with_rht,
        global_scaling=global_scaling,
        encode=encode
    )


@pytest.mark.skipif(not mxfp4_available, reason=reason_for_no_mxfp4)
@pytest.mark.parametrize("in_features, out_features", [(128, 256)])
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1,3], ids=["single_step","multi_step"])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"], ids=["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("with_rht", [True, False], ids=str)
@pytest.mark.parametrize("global_scaling", [False,True], ids=str)
@pytest.mark.parametrize("encode", [False, True], ids=["standard","encode_centric"])
def test_mxfp4_layernorm_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    normalization: str,
    x_dtype: torch.dtype,
    num_steps: int,
    with_rht: bool,
    global_scaling: bool,
    encode: bool
):
    """Test MXFP4 LayerNormLinear module against reference implementation."""
    check_mxfp4_module_versus_reference(
        module_class=te.LayerNormLinear,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        x_dtype=x_dtype,
        num_steps=num_steps,
        with_rht=with_rht,
        global_scaling=global_scaling,
        normalization=normalization,
        encode=encode
    )