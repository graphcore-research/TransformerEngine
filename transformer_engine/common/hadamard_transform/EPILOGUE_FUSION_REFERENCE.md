# Epilogue Fusion Reference

Reference point after rebasing `fp4-custom-quantisation` onto local `main`:

- Rebased branch head: `bc2b70f9aa077e5a4a076efd9884f04abc80db1e`
- Rebase target `main`: `6a68c7336f5ac50b27eeab852965620df5fcee09`

The URL path discussed in review is not present in this checkout. The local source-of-truth analogs are:

- `group_row_cast_col_hadamard_transform_cast_fusion.cu`
- `graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu`
- `customized_pipeline.cuh`

## What NVIDIA is doing in the local TE kernels

The main kernels are:

- `group_row_col_rht_gemm_device`
- `group_row_col_rht_gemm_device_graph_safe`

The important implementation details to preserve as reference are:

- Col epilogue is `TMEM -> registers -> quantize -> global store`, not `TMEM -> shared BF16 bounce -> quantize`.
- The col path uses `cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x`.
- Quantized output uses `make_tiled_copy_D(Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, TD>{}, tiled_t2r)` for 256-bit cache-bypassing stores.
- Pipelines are standardized:
  - `cutlass::PipelineUmmaAsync`
  - `cutlass::PipelineCLCFetchAsync`
  - TE’s customized TMA/UMMA pipeline wrapper in `customized_pipeline.cuh`
- Scheduling is dynamic via Cluster Launch Control, not just a static grid-stride assignment.
- Downcast is hardware-oriented and stochastic:
  - `StochasticNumericConverterBase`
  - `StochasticNumericConverter`
  - `transformer_engine::curanddx::detail::philox4x32_native_state`

## Portable Ideas

These are the pieces most worth borrowing into our kernel work first:

- Remove the col-side shared-memory bounce where possible.
- Keep the hot epilogue register-centric for as long as possible.
- Replace scalar byte-at-a-time stores with packed vector stores when the destination layout allows it.
- Prefer pipeline-style producer/consumer ownership over ad hoc phasebit/semaphore choreography.

## TE-Specific Or Less Portable Pieces

These are useful references, but they depend on TE/CUTLASS structure more heavily:

- `PipelineUmmaAsync` / `PipelineCLCFetchAsync` integration
- Cluster Launch Control tile scheduler
- TE/CuTe tiled copy and TMEM epilogue abstractions

## Important Contract Difference

These TE kernels are not using a pure CTA-local quantization contract.

They explicitly load precomputed global/group amax values:

- `global_a_amax_list`
- `global_d_amax_list`
- shared `global_a_amax[]`
- shared `global_d_amax[]`

That means:

- the epilogue dataflow and store strategy are directly useful
- the global/group `amax` contract is not directly portable to a CTA-local/per-16 NVFP4 design

Any adaptation should call this out explicitly and avoid accidentally introducing a hidden global-`amax` pass.
