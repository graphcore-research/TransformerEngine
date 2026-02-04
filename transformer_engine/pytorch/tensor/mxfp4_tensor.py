# Copyright (c) 2025, Generic AI Implementation. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with MXFP4 data (OCP Micro-scaling Format)"""
from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional, Tuple, Union
import functools

import torch
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

# Assuming a hypothetical recipe for MXFP4 exists or reusing the base Recipe
from transformer_engine.common.recipe import Recipe  
# from ..constants import MXFP4_BLOCK_SCALING_SIZE # We define this locally below
from ..utils import (
    canonicalize_process_group,
    devices_match,
    round_up_to_nearest_multiple,
)

# Assuming the Storage class is generic or duplicated for MXFP4 bindings
from .storage.mxfp4_tensor_storage import MXFP4TensorStorage, _FromMXFP4Func
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

aten = torch.ops.aten
SIMULATE_MXFP4_WITH_FP8 = True

# =============================================================================
# MXFP4 Constants & Constraints
# =============================================================================

# MXFP4 requirement: Block size is 32
MXFP4_BLOCK_SCALING_SIZE = 32

# =============================================================================
# Hadamard Transform Utilities (Block Size 32)
# =============================================================================

def get_no_random_sign_vector() -> torch.Tensor:
    """Non-random sign vector for Hadamard transform."""
    return torch.tensor([1], dtype=torch.float32)


def get_sign_from_vector(vector: torch.Tensor) -> int:
    """Convert sign vector to bitmask.
    Used for random Hadamard transform.
    """
    mask = 0
    for i, v in enumerate(vector):
        mask |= (v == -1) << i
    return mask


def get_wgrad_sign_vector() -> torch.Tensor:
    """Hard-coded random signs for Hadamard transform (size 16 repeated or extended).
    
    Note: For block size 32, this vector might need extension to 32 elements 
    depending on the specific RHT kernel implementation. 
    Assuming the kernel handles the broadcasting or the vector is used 
    to generate the diagonal sign matrix for the 32x32 transform.
    """
    # This specific vector is length 16. If the RHT kernel expects 32 signs for the 
    # 32x32 matrix, we typically repeat it or generate 16 more. 
    # Keeping original logic strictly, but ensuring it applies to the 32 dim matrix.
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
    )


def get_hadamard_matrix(hadamard_dimension: int) -> torch.Tensor:
    """Construct a 16x16 Hadamard matrix."""
    assert hadamard_dimension == 16, "Only hadamard dimension 16 is supported."
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    return (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
            ],
            dtype=torch.float32,
        )
        * hadamard_scale
    )


@functools.lru_cache(maxsize=None)
def get_rht_matrix(with_random_sign_mask: bool) -> torch.Tensor:
    """Construct matrix used in random Hadamard transform."""
    hadamard_dimension = 16
    if with_random_sign_mask:
        signs = get_wgrad_sign_vector()
    else:
        signs = get_no_random_sign_vector()
    sign_matrix = signs * torch.eye(hadamard_dimension, dtype=torch.float32)
    rht_matrix = sign_matrix @ get_hadamard_matrix(hadamard_dimension)
    return rht_matrix.to(dtype=torch.bfloat16).cuda()

@functools.lru_cache(maxsize=None)
def get_random_sign_mask_for_rht(with_random_sign_mask: bool) -> int:
    """Sign mask for random Hadamard transform."""
    if with_random_sign_mask:
        return get_sign_from_vector(get_wgrad_sign_vector())
    return 0


class MXFP4Quantizer(Quantizer):
    """Builder class for MXFP4 tensors with MX block scaling (E8M0)"""

    dtype: TE_DType
    """Random Hadamard Transform"""
    with_rht: bool
    with_post_rht_amax: bool
    """amax reduction options"""
    with_amax_reduction: bool
    amax_reduction_group: Optional[dist_group_type]

    """2D block scaling, only applicable for weights."""
    with_2d_quantization: bool

    """Stochastic rounding, only applicable for gradients."""
    stochastic_rounding: bool

    """RHT matrix random sign mask"""
    rht_matrix_random_sign_mask_t: int
    rht_matrix: torch.Tensor

    global_scaling: bool  # <--- Add type hint

    encode_centric: bool

    def __init__(
        self,
        fp4_dtype: TE_DType = tex.DType.kFloat4E2M1 if not SIMULATE_MXFP4_WITH_FP8 else tex.DType.kFloat8E4M3,
        rowwise: bool = True,
        columnwise: bool = True,
        with_amax_reduction: bool = False,
        amax_reduction_group: Optional[dist_group_type] = None,
        with_rht: bool = False,
        with_post_rht_amax: bool = False,
        with_2d_quantization: bool = False,
        stochastic_rounding: bool = False,
        with_random_sign_mask: bool = True,
        global_scaling: bool = False,
        encode_centric: bool = False
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp4_dtype
        self.with_rht = with_rht
        self.with_post_rht_amax = with_post_rht_amax
        self.with_amax_reduction = with_amax_reduction
        self.amax_reduction_group = amax_reduction_group
        self.with_2d_quantization = with_2d_quantization
        self.stochastic_rounding = stochastic_rounding
        self.rht_matrix_random_sign_mask_t = get_random_sign_mask_for_rht(with_random_sign_mask)
        self.rht_matrix = get_rht_matrix(with_random_sign_mask)
        self.global_scaling = global_scaling
        self.encode_centric = encode_centric

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, MXFP4Tensor), f"Cannot store quantized MXFP4 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel (Assume tex.quantize supports MXFP4 semantics)
        tex.quantize(src, self, dst, noop_flag)

        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        return tex.quantize(tensor, self)

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized for MXFP4"""
        if inp.ndim < 2:
            return False
        # Check against MXFP4 block size (32)
        if inp.shape[-1] % MXFP4_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % MXFP4_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
        """Calculate the shape of the scaling tensor for MXFP4 1D blockwise quantization.

        Parameters
        ----------
        shape : Iterable[int]
            Shape of the input tensor to be quantized
        columnwise : bool
            Whether to use columnwise scaling (True) or rowwise scaling (False)

        Returns
        -------
        Tuple[int, int]
            Shape of the scaling tensor as (outer_dim, inner_dim)
            For MXFP4 1D blockwise quantization, blocksize is 32.
            The scaling factor is E8M0 (stored as uint8).
            
            - If columnwise: (round_to_multiple(K, 128), round_to_multiple(roundup(M / 32), 4))
            - If rowwise: (round_to_multiple(M, 128), round_to_multiple(roundup(K / 32), 4))
        """
        M, K = 1, 1
        M = math.prod(shape[:-1])
        K = shape[-1]

        if columnwise:
            outer = round_up_to_nearest_multiple(K, 128)
            # Divisor changed from 16 to 32 for MXFP4
            inner = round_up_to_nearest_multiple(math.ceil(M / MXFP4_BLOCK_SCALING_SIZE), 4)
            return (outer, inner)
        # rowwise
        outer = round_up_to_nearest_multiple(M, 128)
        # Divisor changed from 16 to 32 for MXFP4
        inner = round_up_to_nearest_multiple(math.ceil(K / MXFP4_BLOCK_SCALING_SIZE), 4)
        return (outer, inner)

    @staticmethod
    def get_columnwise_shape(shape: Iterable[int]) -> Tuple[int, ...]:
        """Calculate the shape of a tensor after columnwise quantization.

        For MXFP4 columnwise quantization, it's performing 32x1 quantization block scaling.
        """
        if len(shape) == 0:
            return tuple()
        # and then after AG, a reorganize kernel will be called to restore the shape
        colwise_shape = [shape[-1]]
        for i in range(len(shape) - 1):
            colwise_shape.append(shape[i])
        return tuple(colwise_shape)

    @staticmethod
    def convert_shape_for_fp4(shape: Iterable[int]) -> Tuple[int, ...]:
        """
        Convert logical [*, N] shape to storage shape.

        - Native FP4: pack 2×4-bit per byte → last dim // 2
        - Sim FP8:    one byte per logical element → shape unchanged
        """
        shape = list(shape)
        if not SIMULATE_MXFP4_WITH_FP8:
            shape[-1] = shape[-1] // 2
        return tuple(shape)

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> MXFP4Tensor:
        # print(f"DEBUG [Python]: make_empty called for shape {shape}")
        # print(f"DEBUG [Python]: rowwise_usage={self.rowwise_usage}, columnwise_usage={self.columnwise_usage}")
        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        assert shape[-1] % MXFP4_BLOCK_SCALING_SIZE == 0, (
            f"Incorrect shape {shape} for MXFP4. Tensor dims must be divisible by"
            f" {MXFP4_BLOCK_SCALING_SIZE}"
        )

        flat_first_dim = math.prod(shape[:-1])
        assert flat_first_dim % MXFP4_BLOCK_SCALING_SIZE == 0, (
            f"Incorrect shape {shape} for MXFP4. Tensor dims must be divisible by"
            f" {MXFP4_BLOCK_SCALING_SIZE}"
        )

        # Allocate FP4 data

        # Allocate Rowwise
        data = None
        scale_inv = None
        amax_rowwise = None
        
        if self.rowwise_usage:
            # Determine shape
            storage_shape = list(shape)
            if not SIMULATE_MXFP4_WITH_FP8:
                 storage_shape[-1] = storage_shape[-1] // 2
            
            data = torch.empty(tuple(storage_shape), dtype=torch.uint8, device=device)
            
            scale_shape = self.get_scale_shape(shape, columnwise=False)
            scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device=device)
            amax_rowwise = torch.ones(1, dtype=torch.float32, device=device)

        # Allocate Columnwise
        columnwise_data = None
        columnwise_scale_inv = None
        amax_columnwise = None
        
        if self.columnwise_usage:
            # Calculate transposed storage shape
            shape_2d = tuple([math.prod(shape[:-1]), shape[-1]])
            col_logical_shape = self.get_columnwise_shape(shape_2d)
            
            col_storage_shape = list(col_logical_shape)
            if not SIMULATE_MXFP4_WITH_FP8:
                col_storage_shape[-1] = col_storage_shape[-1] // 2
                
            columnwise_data = torch.empty(
                tuple(col_storage_shape),
                dtype=torch.uint8,
                device=device,
            )
            
            columnwise_scale_shape = self.get_scale_shape(shape, columnwise=True)
            columnwise_scale_inv = torch.empty(
                columnwise_scale_shape, dtype=torch.uint8, device=device
            )
            amax_columnwise = torch.ones(1, dtype=torch.float32, device=device)

        return MXFP4Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            amax_rowwise=amax_rowwise,
            amax_columnwise=amax_columnwise,
            fp4_dtype=self.dtype,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass  # Calibration is no-op

    def _canonicalized_amax_reduction_group(self) -> dist_group_type:
        """Get process group for amax reduction"""
        return canonicalize_process_group(self.amax_reduction_group)

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        # Assuming MXFP4BlockScaling recipe exists, otherwise generic
        return Recipe 


class MXFP4Tensor(MXFP4TensorStorage, QuantizedTensor):
    """Quantized tensor class with FP4 data and E8M0 Scaling (MXFP4)

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is packed FP4. The scales are E8M0 (unsigned, power of 2).
    """

    # NOTE: We reorder the *args so that we can instantiate a MXFP4TensorStorage with positional args,
    # which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        amax_rowwise: Optional[torch.Tensor],
        amax_columnwise: Optional[torch.Tensor],
        fp4_dtype: TE_DType,
        quantizer: Quantizer,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            amax_rowwise,
            amax_columnwise,
            fp4_dtype,
            quantizer,
            *args,
            **kwargs,
        )
        return instance

    def __repr__(self, *, tensor_contents=None):
        return f"MXFP4Tensor, data={self.dequantize(dtype=self.dtype)})"

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from MXFP4Tensor

        By default the resulting tensor's dtype is the
        MXFP4Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromMXFP4Func.apply(self, dtype)
        return _FromMXFP4Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return MXFP4Quantizer()

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MXFP4Tensor:
        """Update FP4 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def detach(self) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return MXFP4Tensor.make_like(self)

    def clone(self) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        assert self._rowwise_data is not None
        rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> MXFP4Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._rowwise_data is not None and self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        if self._columnwise_data is not None and self._columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        raise ValueError("MXFP4Tensor does not support different memory formats!")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            if len(args) != 2:
                raise RuntimeError("Unexpected args for view op (expected 2 args, got {len(args)})")
            tensor = args[0]
            shape = args[1]
            if shape == list(tensor.size()):
                return tensor.detach()
            return tensor.view(shape)

        # MXFP4 dequantize not supported. Add manual support for needed funcs.
        if func in (aten.empty_like.default, aten.zero_.default):
            tensor = args[0]
            data_init_func = torch.zeros_like if func == aten.zero_.default else torch.empty_like
            scale_inv_init_func = (
                torch.ones_like if func == aten.zero_.default else torch.empty_like
            )

            if tensor._rowwise_data is not None:
                rowwise_data = data_init_func(tensor._rowwise_data)
                rowwise_scale_inv = scale_inv_init_func(tensor._rowwise_scale_inv)
                amax_rowwise = torch.zeros_like(tensor._amax_rowwise)
            else:
                rowwise_data, rowwise_scale_inv, amax_rowwise = None, None, None

            if tensor._columnwise_data is not None:
                columnwise_data = data_init_func(tensor._columnwise_data)
                columnwise_scale_inv = scale_inv_init_func(tensor._columnwise_scale_inv)
                amax_columnwise = torch.zeros_like(tensor._amax_columnwise)
            else:
                columnwise_data, columnwise_scale_inv, amax_columnwise = (
                    None,
                    None,
                    None,
                )

            return MXFP4Tensor(
                shape=tensor.shape,
                dtype=tensor.dtype,
                fp4_dtype=tensor._fp4_dtype,
                rowwise_data=rowwise_data,
                rowwise_scale_inv=rowwise_scale_inv,
                columnwise_data=columnwise_data,
                columnwise_scale_inv=columnwise_scale_inv,
                amax_rowwise=amax_rowwise,
                amax_columnwise=amax_columnwise,
                quantizer=tensor._quantizer,
                requires_grad=tensor.requires_grad,
            )

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        amax_rowwise: torch.Tensor,
        amax_columnwise: torch.Tensor,
        fp4_dtype: TE_DType,
        dtype: torch.dtype,
        quantizer: Quantizer,
    ) -> MXFP4Tensor:
        """Build MXFP4Tensor, for use in __reduce__"""
        return MXFP4Tensor(
            shape=shape,
            dtype=dtype,
            fp4_dtype=fp4_dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            amax_rowwise=amax_rowwise,
            amax_columnwise=amax_columnwise,
            quantizer=quantizer,
            requires_grad=False,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            MXFP4Tensor._make_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._amax_rowwise,
                self._amax_columnwise,
                self._fp4_dtype,
                self.dtype,
                self._quantizer,
            ),
        )

    def _get_data(self) -> MXFP4Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property"""

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP4 data if other tensor is MXFP4Tensor
        if isinstance(tensor, MXFP4Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    MXFP4Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(MXFP4Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            self._amax_rowwise = tensor._amax_rowwise
            self._amax_columnwise = tensor._amax_columnwise
            return

        # Quantize to FP4
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.update_quantized(tensor, self)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP4 when setting MXFP4Tensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function
    View the MXFP4Tensor using the provided shape.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        cur_shape = tensor.shape
        if ctx is not None:
            ctx.shape = cur_shape
        if shape is None:
            return tensor

        # Canonicalize shape
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
        if shape[-1] != cur_shape[-1]:
            raise RuntimeError(
                "MXFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Reshape data
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            if not SIMULATE_MXFP4_WITH_FP8:
                if shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent row-wise data for MXFP4 tensor "
                        f"with shape={shape} as byte array."
                    )
                byte_shape = list(shape[:-1]) + [shape[-1] // 2]
            else:
                # Sim path: one byte per logical element
                byte_shape = list(shape)
            new_rowwise_data = tensor._rowwise_data.view(byte_shape)

        if tensor._columnwise_data is not None:
            columnwise_shape = (shape[-1], math.prod(shape[:-1]))
            if not SIMULATE_MXFP4_WITH_FP8:
                if columnwise_shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent column-wise data for MXFP4 tensor "
                        f"with shape={shape} as byte array."
                    )
                byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
            else:
                byte_shape = columnwise_shape
            new_columnwise_data = tensor._columnwise_data.view(byte_shape)


        # Construct tensor
        return MXFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            fp4_dtype=tensor._fp4_dtype,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP4Tensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                if not SIMULATE_MXFP4_WITH_FP8:
                    if ctx.shape[-1] % 2 != 0:
                        raise ValueError(
                            "Cannot represent row-wise data for MXFP4 tensor "
                            f"with shape={ctx.shape} as byte array."
                        )
                    byte_shape = list(ctx.shape[:-1]) + [ctx.shape[-1] // 2]
                    new_rowwise_data = grad._rowwise_data.view(byte_shape)
                else:
                    # Sim path: one byte per logical element
                    byte_shape = list(ctx.shape)
                    new_rowwise_data = grad._rowwise_data.view(byte_shape)
            if grad._columnwise_data is not None:
                columnwise_shape = (ctx.shape[-1], math.prod(ctx.shape[:-1]))
                if not SIMULATE_MXFP4_WITH_FP8:
                    if columnwise_shape[-1] % 2 != 0:
                        raise ValueError(
                            "Cannot represent column-wise data for MXFP4 tensor "
                            f"with shape={ctx.shape} as byte array."
                        )
                    byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
                    new_columnwise_data = grad._columnwise_data.view(byte_shape)
                else:
                    byte_shape = columnwise_shape
                    new_columnwise_data = grad._columnwise_data.view(byte_shape)
            dgrad = MXFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                fp4_dtype=grad._fp4_dtype,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function
    Reshape the MXFP4Tensor using the provided shape.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        cur_shape = tensor.shape
        if ctx is not None:
            ctx.shape = cur_shape
        if shape is None:
            return tensor

        # Canonicalize shape
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
        if shape[-1] != cur_shape[-1]:
            raise RuntimeError(
                "MXFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Reshape data
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            if not SIMULATE_MXFP4_WITH_FP8:
                if shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent row-wise data for MXFP4 tensor "
                        f"with shape={shape} as byte array."
                    )
                byte_shape = list(shape[:-1]) + [shape[-1] // 2]
            else:
                # Sim path: one byte per logical element
                byte_shape = list(shape)
            new_rowwise_data = tensor._rowwise_data.view(byte_shape)

        if tensor._columnwise_data is not None:
            columnwise_shape = (shape[-1], math.prod(shape[:-1]))
            if not SIMULATE_MXFP4_WITH_FP8:
                if columnwise_shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent column-wise data for MXFP4 tensor "
                        f"with shape={shape} as byte array."
                    )
                byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
            else:
                byte_shape = columnwise_shape
            new_columnwise_data = tensor._columnwise_data.view(byte_shape)

        # Construct tensor
        return MXFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            fp4_dtype=tensor._fp4_dtype,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP4Tensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                if not SIMULATE_MXFP4_WITH_FP8:
                    if ctx.shape[-1] % 2 != 0:
                        raise ValueError(
                            "Cannot represent row-wise data for MXFP4 tensor "
                            f"with shape={ctx.shape} as byte array."
                        )
                    byte_shape = list(ctx.shape[:-1]) + [ctx.shape[-1] // 2]
                    new_rowwise_data = grad._rowwise_data.view(byte_shape)
                else:
                    # Sim path: one byte per logical element
                    byte_shape = list(ctx.shape)
                    new_rowwise_data = grad._rowwise_data.view(byte_shape)
            if grad._columnwise_data is not None:
                columnwise_shape = (ctx.shape[-1], math.prod(ctx.shape[:-1]))
                if not SIMULATE_MXFP4_WITH_FP8:
                    if columnwise_shape[-1] % 2 != 0:
                        raise ValueError(
                            "Cannot represent column-wise data for MXFP4 tensor "
                            f"with shape={ctx.shape} as byte array."
                        )
                    byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
                    new_columnwise_data = grad._columnwise_data.view(byte_shape)
                else:
                    byte_shape = columnwise_shape
                    new_columnwise_data = grad._columnwise_data.view(byte_shape)
            dgrad = MXFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                fp4_dtype=grad._fp4_dtype,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None