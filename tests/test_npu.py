"""On-device contract tests for TopKPerRow (requires an Ascend NPU + CANN).

Run on the target machine after building the custom op into ``torch.ops._C_ascend``:

    python -m pytest tests/test_npu.py -v

The CPU reference in ``tests/reference.py`` is the ground truth.  These tests
are skipped automatically when torch_npu / the custom op are unavailable.
"""

from __future__ import annotations

import unittest

import numpy as np

try:  # the whole module is a no-op off-device
    import pytest
    import torch
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"requires pytest + torch + torch_npu: {exc}") from exc

from reference import TOPK_MAX, check_invariants, top_k_per_row_numpy

try:  # the operator is registered by vllm_ascend.utils.enable_custom_op()
    import vllm_ascend  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
except Exception:  # pragma: no cover - standalone builds
    pass


def _op():
    for ns in ("_C_ascend", "_C"):
        try:
            return getattr(torch.ops, ns).top_k_per_row_prefill
        except AttributeError:
            continue
    pytest.skip("top_k_per_row is not registered in this build")


def _run(logits, starts, ends, topk):
    fn = _op()
    num_rows = logits.shape[0]
    out = torch.empty((num_rows, topk), dtype=torch.int32, device=logits.device)
    fn(logits, starts, ends, out, num_rows, logits.stride(0), logits.stride(1), topk)
    return out.cpu().numpy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 32, 512, 2048])
def test_matches_cpu_reference(dtype, topk):
    rng = np.random.default_rng(1234 + topk)
    rows, width = 8, 4096
    base = rng.normal(size=(rows, width)).astype(np.float32)
    starts, ends = [], []
    for _ in range(rows):
        s = int(rng.integers(0, width))
        e = int(rng.integers(s, width + 1))
        starts.append(s)
        ends.append(e)
    starts = np.asarray(starts, dtype=np.int32)
    ends = np.asarray(ends, dtype=np.int32)

    logits = torch.from_numpy(base).to("npu:0").to(dtype)
    s_t = torch.from_numpy(starts).to("npu:0")
    e_t = torch.from_numpy(ends).to("npu:0")
    got = _run(logits, s_t, e_t, topk)

    # compare on the value the device actually saw (bf16 rounded)
    seen = logits.float().cpu().numpy()
    want = top_k_per_row_numpy(seen, starts, ends, topk)
    np.testing.assert_array_equal(got, want)
    assert check_invariants(seen, starts, ends, topk, got) == []


def test_empty_and_short_windows():
    width = 1024
    rng = np.random.default_rng(7)
    base = rng.normal(size=(4, width)).astype(np.float32)
    starts = np.array([0, 10, 900, 512], dtype=np.int32)
    ends = np.array([0, 10, 1024, 520], dtype=np.int32)  # lengths 0,0,124,8
    topk = 64
    logits = torch.from_numpy(base).to("npu:0")
    got = _run(logits, torch.from_numpy(starts).to("npu:0"), torch.from_numpy(ends).to("npu:0"), topk)
    want = top_k_per_row_numpy(base, starts, ends, topk)
    np.testing.assert_array_equal(got, want)


def test_decode_entry_point():
    width = 2048
    rng = np.random.default_rng(11)
    base = rng.normal(size=(6, width)).astype(np.float32)
    seq_lens = torch.tensor([1, 7, 64, 1234, 2048, 500], dtype=torch.int32, device="npu:0")
    topk = 128
    logits = torch.from_numpy(base).to("npu:0")
    out = torch.empty((6, topk), dtype=torch.int32, device="npu:0")
    fn = getattr(torch.ops, "_C_ascend").top_k_per_row_decode
    fn(logits, 1, seq_lens, out, 6, logits.stride(0), logits.stride(1), topk)
    got = out.cpu().numpy()

    starts = np.zeros(6, dtype=np.int32)
    ends = seq_lens.cpu().numpy().astype(np.int32)
    want = top_k_per_row_numpy(base, starts, ends, topk)
    np.testing.assert_array_equal(got, want)


def test_decode_entry_point_next_n():
    """``seq_lens`` is one length per *request*; ``next_n`` rows expand from each.

    Two requests x next_n=3 gives 6 rows, so ``numel()`` (2) deliberately differs
    from the row count (6): the shape a ``numel() / next_n`` batch computation
    would get backwards.
    """
    width = 512
    next_n = 3
    topk = 64
    rng = np.random.default_rng(13)
    seq_lens_host = np.array([64, width], dtype=np.int32)  # 2 requests -> 6 rows
    num_rows = seq_lens_host.size * next_n
    base = rng.normal(size=(num_rows, width)).astype(np.float32)

    offsets = np.arange(next_n, dtype=np.int32)
    ends = seq_lens_host.reshape(-1, 1) - next_n + 1 + offsets.reshape(1, next_n)
    ends = np.clip(ends, 0, width).reshape(-1).astype(np.int32)
    starts = np.zeros(num_rows, dtype=np.int32)

    logits = torch.from_numpy(base).to("npu:0")
    seq_lens = torch.from_numpy(seq_lens_host).to("npu:0")
    out = torch.empty((num_rows, topk), dtype=torch.int32, device="npu:0")
    fn = getattr(torch.ops, "_C_ascend").top_k_per_row_decode
    fn(logits, next_n, seq_lens, out, num_rows,
       logits.stride(0), logits.stride(1), topk)
    got = out.cpu().numpy()

    want = top_k_per_row_numpy(base, starts, ends, topk)
    np.testing.assert_array_equal(got, want)


def test_topk_upper_bound_guard():
    logits = torch.zeros((2, 64), dtype=torch.float32, device="npu:0")
    with pytest.raises(RuntimeError):
        _run(logits, torch.zeros(2, dtype=torch.int32, device="npu:0"),
             torch.full((2,), 64, dtype=torch.int32, device="npu:0"), TOPK_MAX + 1)
