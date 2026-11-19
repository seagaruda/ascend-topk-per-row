# top-k-per-row (Ascend C)

A CANN / Ascend C implementation of `top_k_per_row` — the ragged per-row top-k
selection kernel that vLLM ships only for CUDA — packaged in the exact layout
`vllm-ascend` uses for its custom aclnn operators (`csrc/<op>/{op_host,op_kernel}`
+ `torch.ops._C_ascend` bindings).

```
logits (num_rows, width)  ──┐
row_starts (num_rows,) int32 ├─►  TopKPerRow  ──►  indices (num_rows, topk) int32
row_ends   (num_rows,) int32 ┘                     (descending, -1 padded)
```

* **Status:** reference semantics + CPU test-suite green (8/8). The Ascend C
  kernel is written against the CANN 8.x `AscendC::TopK` high-level API and the
  host tiling probes it with `TopKTilingFunc` / `GetTopKMaxMinTmpSize`, but it
  has **not yet been compiled on an NPU** in this repository (no Ascend device
  available where it was authored). See [Verification status](#verification-status).
* **License:** Apache-2.0 here. If this operator is merged into `vllm-ascend/csrc/`, switch the
  header to CANN Open Software License Agreement V2.0 like its siblings.

---

## Why this operator

`top_k_per_row` is the top-k step of DeepSeek's sparse-attention indexer (DSA):
every layer, every decode step, for every row, pick the `topk` most relevant
past tokens out of a window that is ragged across rows.

| Evidence | Result |
|---|---|
| `vllm/csrc/topk.cu` | 324 lines, header comment: *"Persistent TopK kernel for DeepSeek V3 sparse attention indexer"* |
| `vllm/csrc/ops.h` | `top_k_per_row_prefill(...)`, `top_k_per_row_decode(...)` — two CUDA-only entries |
| `vllm/model_executor/layers/indexer_topk.py` | five CUDA backends (`deep_select`, `cooperative`, `persistent`, `flashinfer`, `flashinfer_gvr2`), all CUDA-only |
| GitHub code search `repo:vllm-project/vllm-ascend top_k_per_row` | **0 hits** (2026-11-15) |
| Same search for `hadacore_transform` / `vocab_parallel_embedding` | 26 / 35 hits — the search is not silently broken |
| CANN `aclnnTopK` | last-dim only, fixed `k`, returns values+indices, no ragged window, no `-1` padding → cannot express this op |

So on Ascend today the only path is a CPU fallback (`sort` the whole padded row
in PyTorch, materialise a mask, gather), which is O(W log W) per row plus a large
temporary. That is what this operator removes.

## Semantics (the contract)

For row `r`:

```
s = clamp(row_starts[r], 0, width)
e = clamp(row_ends[r],   0, width)
L = max(0, e - s)
```

1. `L >= topk` → the row holds `topk` column indices from `[s, e)`.
2. `L <  topk` → the first `L` slots hold the sorted valid indices, the rest are `-1`.
3. Ordering: value descending; **ties resolve to the smaller column index**
   (deterministic — the CUDA kernel makes no such promise, this op does, so
   tests are stable).
4. `NaN` ranks below every finite value and below `-inf`; `+0.0 == -0.0`.
5. `logits` is compared in fp32 (fp16/bf16 inputs are cast in the kernel).
6. `topk ∈ (0, 2048]`.

`tests/reference.py` is the executable version of the above (two independent
implementations, cross-checked), `tests/test_semantics.py` locks it in.

## Implementation

```
csrc/top_k_per_row/
├── op_host/
│   ├── top_k_per_row_def.cpp          # OpDef: logits/row_starts/row_ends -> indices, attr topk
│   ├── top_k_per_row_infershape.cpp   # out shape (num_rows, topk), int32
│   ├── top_k_per_row_tiling.h/.cpp    # UB budget + TopK tiling probing
│   ├── CMakeLists.txt
│   └── op_api/aclnn_top_k_per_row.*   # aclnn two-phase entry
├── op_kernel/
│   ├── top_k_per_row.cpp              # kernel entry, dispatch on dtype tiling key
│   └── top_k_per_row.h                # KernelTopKPerRow
└── top_k_per_row_torch_adpt.h         # top_k_per_row_prefill / _decode wrappers
```

Per-core work is row-parallel (`rows_per_core = ceil(num_rows / num_aiv_cores)`).
Inside a row:

1. the window `[s, e)` is streamed in `chunk_size` blocks; the whole chunk is
   pre-filled with the `-FLT_MAX` sentinel by `Duplicate` and only the valid
   prefix is copied on top. The copy's `paddingCount` is a `uint8_t`, so it
   silently drops a tail longer than 255 elements (`chunk=32768, L=256` keeps 0
   of 32512 pads) — pre-filling makes the tail independent of its length;
2. each chunk is reduced to its own top-k with `AscendC::TopK`
   (`TopKMode::TOPK_NORMAL`), then one `Adds` turns block-local positions into
   absolute column indices;
3. blocks are merged `max_chunks` at a time into a running candidate list of
   `topk_pad` slots — the merge is another `TopK` over a
   `(1 + max_chunks) * topk_pad` row; its output is a *position* inside the
   concatenated value buffer, so a single `Gather` maps positions back to
   column indices kept in a layout-identical index buffer (no scalar loops);
4. the tail past `min(topk, L)` is `Duplicate`d with `-1` and written with
   `DataCopyPad` (byte-granular, so `topk` need not be 32-aligned).

Tiling never looks at tensor *values*, only shapes/attrs, so the op is safe for
graph capture (ACLGraph). `chunk_size`, `max_chunks` and the TopK tmp size are
chosen on the host by probing `AscendC::TopKTilingFunc` and
`AscendC::GetTopKMaxMinTmpSize` against the UB budget. The physical chunk is
`CeilAlign(min(width, 32768), 32)` with a floor of `max(32, topk_pad)`, so rows
narrower than 256 columns — down to a single column — are legal input rather
than rejected; the loader pads them up.

## Integrating into vllm-ascend

1. copy `csrc/top_k_per_row/` to `vllm-ascend/csrc/attention/top_k_per_row/`;
2. add `top_k_per_row` to `CUSTOM_OPS` in `csrc/build_aclnn.sh`;
3. include `top_k_per_row_torch_adpt.h` from `csrc/torch_binding.cpp` and bind:

```cpp
void top_k_per_row_prefill(const at::Tensor& logits, const at::Tensor& ks,
                           const at::Tensor& ke, at::Tensor& indices, int64_t num_rows,
                           int64_t stride0, int64_t stride1, int64_t topk)
{
    vllm_ascend::top_k_per_row_prefill(logits, ks, ke, indices, num_rows, stride0, stride1, topk);
}
```

4. add a meta implementation in `csrc/torch_binding_meta.cpp` so the op can be
   captured into an ACLGraph (`indices` shape `(num_rows, topk)`, `-1` filled);
5. wire `vllm/model_executor/layers/indexer_topk.py`'s `per_row` path to this op
   when `current_platform.is_ascend()`.

The kernel itself does not depend on vLLM: it only needs aclnn, so it can also be
called straight from aclnn or from a TileLang/Triton reference harness.

## Verification status

| Layer | Status |
|---|---|
| CPU contract tests (`tests/test_semantics.py`) | **passing** — 9 tests, 50 random ragged shapes + ties + NaN/±inf + clamping + decode row expansion |
| Host tiling logic | reviewed, not executed (needs `PlatformAscendC`) |
| Ascend C kernel | **not compiled yet** — needs CANN 8.3+ on A2/A3 |
| End-to-end vs `torch.topk` | `tests/test_npu.py`, run on device |

`tests/test_semantics.py` runs anywhere:

```bash
python3 -m unittest discover -s tests -v
```

To validate the kernel on a real machine:

```bash
# 1. build inside a vllm-ascend tree (CANN 8.3.RC1+, A2/A3)
cd vllm-ascend && bash csrc/build_aclnn.sh
# 2. run the on-device contract tests
VLLM_ASCEND_ENABLE_CUSTOM_OP=1 python -m pytest /path/to/this/repo/tests/test_npu.py -v
```

Two places deserve a careful review on first build, because they are the only
API surfaces not exercised by the CPU tests:

* `AscendC::TopK` template/deduction (module signature mirrors
  `xLLM-AI/xllm-ops` `moe_fused_add_topk` and `hicann/ops-transformer`
  `moe_fused_topk`, which compile on A2/A3);
* `Gather(dst, src, byteOffsets, baseAddr, count)` — the 5-argument vector form,
  offsets in **bytes** (as used by `Ascend/DrivingSDK`'s
  `kernels/select_idx_with_mask`).

## Performance expectations

The kernel reads each window exactly once (per chunk) and never materialises a
mask, so it is O(W) memory traffic + `O(chunks · k log k)` selection work, versus
the fallback's O(W log W) full sort plus mask allocation. For the DSA shapes that
motivate it (`topk ∈ {512, 1024, 2048}`, `W = 32K … 128K`, `rows = batch`), the
fallback is a sort of the full padded row per layer per step; this op turns that
into a streaming select. Absolute numbers still have to be measured on device —
no benchmark numbers are claimed here.

Roadmap: (a) `Sort32` + `MrgSort4` merge instead of repeated `TopK` calls to cut
the merge passes; (b) a decode-specialised path that keeps the running candidate
list resident across layers; (c) integrate as the `per_row` backend in
vllm-ascend's `indexer_topk.py` so no Python-level dispatch is needed.

## Related notes

* `docs/tilelang-ascend-notes.md` — what [tile-ai/tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)
  is, and where a TileLang version of this op would fit.
* `docs/gap-analysis.md` — the full CUDA-vs-CANN gap sweep this op was picked from.
