import torch
import gfloat
from gfloat.types import FormatInfo, Domain
from low_bits_training.quantization.customTorchQuantiser import round_ndarray

from dataclasses import dataclass

@dataclass
class LiteFormatInfo:
    """Lightweight struct to mimic FormatInfo for compiled kernel, avoiding gfloat logic."""
    name: str
    max: float
    min: float
    precision: int
    bias: int
    has_subnormals: bool
    is_signed: bool
    has_nz: bool
    num_high_nans: int
    k: int
    is_twos_complement: bool
    domain: Domain
    
    # helper property for round_ndarray if it uses num_nans
    @property
    def num_nans(self):
        return self.num_high_nans # Simplified assumption

# -------------------------------------------------------------------------
# COMPILED QUANTIZATION KERNEL (Fusion Optimizer)
# -------------------------------------------------------------------------
# @torch.compile(dynamic=True) # REMOVED: User suggestion + CUDAGraphs error
def _compiled_quantize_core(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    tile_len_x: int,
    tile_len_y: int,
    using_2d_quantization: bool,
    scale_max_val: float,
    data_max_val: float,
    encode_centric: bool,
    use_global_scale: bool,
    scale_round_mode_str: str,
    round_mode_str: str,
    # Data Format Primitives
    data_fmt_max: float,
    data_fmt_min: float,
    data_fmt_precision: int,
    data_fmt_bias: int,
    data_fmt_has_subnormals: bool,
    data_fmt_is_signed: bool,
    data_fmt_has_nz: bool,
    data_fmt_k: int,
    data_fmt_is_twos_complement: bool,
    # Scale Format Primitives
    scale_fmt_max: float,
    scale_fmt_min: float,
    scale_fmt_precision: int,
    scale_fmt_bias: int,
    scale_fmt_has_subnormals: bool,
    scale_fmt_is_signed: bool,
    scale_fmt_has_nz: bool,
    scale_fmt_k: int,
    scale_fmt_is_twos_complement: bool,
    # Defaults at end
    data_fmt_num_high_nans: int = 0,
    scale_fmt_num_high_nans: int = 0
):
    # Construct LiteFormatInfo for Data
    fi_data = LiteFormatInfo(
        name="data_fmt",
        max=data_fmt_max,
        min=data_fmt_min,
        precision=data_fmt_precision,
        bias=data_fmt_bias,
        has_subnormals=data_fmt_has_subnormals,
        is_signed=data_fmt_is_signed,
        has_nz=data_fmt_has_nz,
        num_high_nans=data_fmt_num_high_nans,
        k=data_fmt_k,
        is_twos_complement=data_fmt_is_twos_complement,
        domain=Domain.Finite
    )
    
    # Construct LiteFormatInfo for Scale
    fi_scale = LiteFormatInfo(
        name="scale_fmt",
        max=scale_fmt_max,
        min=scale_fmt_min,
        precision=scale_fmt_precision,
        bias=scale_fmt_bias,
        has_subnormals=scale_fmt_has_subnormals,
        is_signed=scale_fmt_is_signed,
        has_nz=scale_fmt_has_nz,
        num_high_nans=scale_fmt_num_high_nans,
        k=scale_fmt_k,
        is_twos_complement=scale_fmt_is_twos_complement,
        domain=Domain.Finite
    )

    m, n = x.shape
    
    SCALE_MAX = torch.tensor(scale_max_val, device=x.device, dtype=torch.float32)
    DATA_MAX = torch.tensor(data_max_val, device=x.device, dtype=torch.float32)
    
    # 2. Reshape and Calculate Block Max (vec_max)
    # Using Unified View Logic regarding dimensions for Broadcasting
    grid_y = m // tile_len_y
    grid_x = n // tile_len_x
    
    if using_2d_quantization:
        # 2D Path: Unfold for Max Calculation (per User preference/verified)
        x_blocks = (
            x.unfold(0, tile_len_y, tile_len_y)
            .unfold(1, tile_len_x, tile_len_x)
            .to(torch.float32)
        )
        block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))
        # block_amax: (grid_y, grid_x)
        
        # Keep vec_max small! (grid_y, grid_x, 1)
        vec_max = block_amax.unsqueeze(-1)
        
        # View x for broadcasting: (grid_y, tile_y, grid_x, block_x)
        # Note: x is passed as (M, N). We view it to separate tile dimensions.
        x_for_mul = x.view(grid_y, tile_len_y, grid_x, tile_len_x)
        
    else:
        # 1D Path: Row-wise chunks
        x_reshaped = x.view(m, grid_x, tile_len_x)
        vec_max_full = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(torch.float32)
        # vec_max_full: (M, grid_x, 1). M = grid_y * 1.
        
        vec_max = vec_max_full
        x_for_mul = x_reshaped # (M, grid_x, block_x)

    # 3. Handle Scaling Strategy
    # Operations are performed on 'vec_max' which might be small (2D) or full (1D)
    
    if encode_centric:
        # === ENCODE CENTRIC PATH (Generalized) ===
        max_f32 = torch.tensor(torch.finfo(torch.float32).max, device=x.device, dtype=torch.float32)
        one = torch.tensor(1.0, device=x.device, dtype=torch.float32)

        # 1. Local scale
        decode_scale = torch.div(vec_max, DATA_MAX)
        
        # 2. Compute global encode scale
        global_encode_scale = torch.div(SCALE_MAX * DATA_MAX, global_amax)
        global_encode_scale = torch.min(global_encode_scale, max_f32)
        
        global_encode_scale = torch.where(
            (global_amax == 0.0) | (global_encode_scale == 0.0),
            one,
            global_encode_scale
        )
        
        global_decode_scale = 1.0 / global_encode_scale
        
        # 3. Combine Scales
        decode_scale = decode_scale * global_encode_scale
        
        # 4. Clamp
        decode_scale = torch.clamp(decode_scale, min=-SCALE_MAX, max=SCALE_MAX)
        scale_for_zeros = (one / SCALE_MAX)
        decode_scale = torch.where(vec_max <= 1e-9, scale_for_zeros, decode_scale)

        # Rounding (Optimized: runs on small tensor if 2D)
        scale_round_mode = getattr(gfloat.RoundMode, scale_round_mode_str, gfloat.RoundMode.TiesToEven)
        if scale_round_mode==gfloat.RoundMode.Stochastic:
            srbits = torch.randint(0, 65536, vec_max.shape, device=x.device, dtype=torch.int32)
            decode_scale_f32 = round_ndarray(fi_scale, decode_scale, scale_round_mode, sat=True, srbits=srbits, srnumbits=16)
        else:
            decode_scale_f32 = round_ndarray(fi_scale, decode_scale, scale_round_mode, sat=True)
        
        decode_scale = decode_scale_f32

        # 5. Calculate Encode Scale
        encode_scale_normal = 1.0 / (decode_scale * global_decode_scale)
        encode_scale_zeros = SCALE_MAX * global_encode_scale
        
        encode_scale = torch.where(vec_max <= 1e-9, encode_scale_zeros, encode_scale_normal)
        encode_scale = torch.min(encode_scale, max_f32)
        
    else:
        # === DECODE CENTRIC PATH (Generalized) ===
        decode_scale = torch.div(vec_max, DATA_MAX)
        if use_global_scale:
            global_encode_scale = torch.div(SCALE_MAX * DATA_MAX, global_amax)
            global_encode_scale = torch.min(global_encode_scale, torch.tensor(torch.finfo(torch.float32).max, device=x.device))
            global_encode_scale = torch.where(global_encode_scale ==0, torch.tensor(1.0, device=x.device), global_encode_scale)
            
            global_decode_scale = 1.0 / global_encode_scale
            decode_scale = decode_scale * global_encode_scale
        else:
            global_decode_scale = 1.0

        decode_scale = torch.clamp(decode_scale, min=-SCALE_MAX, max=SCALE_MAX)
        
        scale_round_mode = getattr(gfloat.RoundMode, scale_round_mode_str, gfloat.RoundMode.TiesToEven)
        if scale_round_mode==gfloat.RoundMode.Stochastic:
            srbits = torch.randint(0, 65536, vec_max.shape, device=x.device, dtype=torch.int32)
            decode_scale_f32 = round_ndarray(fi_scale, decode_scale, scale_round_mode, sat=True, srbits=srbits, srnumbits=16)
        else:
            decode_scale_f32 = round_ndarray(fi_scale, decode_scale, scale_round_mode, sat=True)
        
        decode_scale = decode_scale_f32
        
        encode_scale = 1.0 / (decode_scale_f32 * global_decode_scale)
        max_float = torch.tensor(torch.finfo(torch.float32).max, device=x.device)
        encode_scale = torch.min(encode_scale, max_float)

    # Apply Scaling with Broadcasting
    if using_2d_quantization:
        # encode_scale: (grid_y, grid_x, 1)
        # x_for_mul: (grid_y, tile_y, grid_x, block_x)
        # Broadcast: unsqueeze dim 1
        encode_scale_expanded = encode_scale.unsqueeze(1)
        scaled_x = x_for_mul.to(torch.float32) * encode_scale_expanded
    else:
        # encode_scale: (M, grid_x, 1)
        # x_for_mul: (M, grid_x, block_x)
        scaled_x = x_for_mul.to(torch.float32) * encode_scale

    # Saturate and Reshape back to flat (M, N) for rounding
    # Note: round_ndarray element-wise, shape doesn't matter much, 
    # but we need contiguous behavior? Clip preserves, Reshape preserves.
    clipped_x = torch.clamp(scaled_x, -DATA_MAX, DATA_MAX).reshape(m, n)

    
    # --- Quantize Data (using environment strategy) ---
    round_mode = getattr(gfloat.RoundMode, round_mode_str, gfloat.RoundMode.TiesToEven)

    if round_mode == gfloat.RoundMode.Stochastic:
        srbits = torch.randint(0, 65536, x.shape, device=x.device, dtype=torch.int32)
        quantized = round_ndarray(
            fi_data, clipped_x, round_mode, sat=True, srbits=srbits, srnumbits=16
        )
    else:
        # Use fi_data (replaces format_info_ocp_e2m1)
        quantized = round_ndarray(fi_data, clipped_x, round_mode, sat=True)
        
    if using_2d_quantization:
        # Expand Y dimension back to M to match caller expectation (sx needs to be M x grid_x)
        decode_scale_expanded = decode_scale.unsqueeze(1).expand(grid_y, tile_len_y, grid_x, 1).reshape(m, grid_x, 1)
        return quantized, decode_scale_expanded.squeeze(-1)
    else:
        return quantized, decode_scale.squeeze(-1)
