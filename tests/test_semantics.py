"""CPU-side contract tests for TopKPerRow. Runnable with plain unittest:

    python3 -m unittest discover -s tests -v

No torch / CANN required: this pins down the *specification* that the device
kernels must match, and cross-checks the two independent reference
implementations.
"""

from __future__ import annotations

import unittest

import numpy as np

from reference import (
    TOPK_MAX,
    check_invariants,
    top_k_per_row_loop,
    top_k_per_row_numpy,
)


def _ragged(num_rows, width, rng, empty_rows=0):
    starts = np.zeros(num_rows, dtype=np.int32)
    ends = np.zeros(num_rows, dtype=np.int32)
    for r in range(num_rows):
        s = int(rng.integers(0, width))
        e = int(rng.integers(s, width + 1))
        starts[r] = s
        ends[r] = e
    for r in rng.choice(num_rows, size=min(empty_rows, num_rows), replace=False):
        starts[r] = ends[r] = int(rng.integers(0, width + 1))
    return starts, ends


class TestReferenceEquivalence(unittest.TestCase):
    """The two reference implementations must agree everywhere."""

    def test_random_ragged(self):
        rng = np.random.default_rng(0)
        for trial in range(50):
            rows = int(rng.integers(1, 9))
            width = int(rng.integers(1, 300))
            topk = int(rng.integers(1, min(TOPK_MAX, width) + 1))
            logits = rng.normal(size=(rows, width)).astype(np.float32)
            s, e = _ragged(rows, width, rng, empty_rows=1)
            a = top_k_per_row_numpy(logits, s, e, topk)
            b = top_k_per_row_loop(logits, s, e, topk)
            np.testing.assert_array_equal(a, b, err_msg=f"trial {trial}")
            self.assertEqual(check_invariants(logits, s, e, topk, a), [])

    def test_heavy_ties(self):
        rng = np.random.default_rng(1)
        logits = rng.integers(0, 4, size=(6, 200)).astype(np.float32)
        s = np.zeros(6, dtype=np.int32)
        e = np.full(6, 200, dtype=np.int32)
        for topk in (1, 7, 32, 64, 199, 200):
            a = top_k_per_row_numpy(logits, s, e, topk)
            b = top_k_per_row_loop(logits, s, e, topk)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(check_invariants(logits, s, e, topk, a), [])
            # ties must resolve to ascending column index
            for r in range(6):
                vals = logits[r, a[r]]
                self.assertTrue(np.all(np.diff(vals) <= 0))
                for i in range(topk - 1):
                    if vals[i] == vals[i + 1]:
                        self.assertLess(a[r, i], a[r, i + 1])

    def test_special_values(self):
        logits = np.array(
            [
                [np.nan, 1.0, -np.inf, np.inf, 0.0, -0.0],
                [-0.0, 0.0, np.nan, np.nan, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        s = np.zeros(2, dtype=np.int32)
        e = np.full(2, 6, dtype=np.int32)
        for topk in (1, 3, 6):
            a = top_k_per_row_numpy(logits, s, e, topk)
            b = top_k_per_row_loop(logits, s, e, topk)
            np.testing.assert_array_equal(a, b)
        out = top_k_per_row_numpy(logits, s, e, 6)
        # inf first, NaN last
        self.assertEqual(out[0, 0], 3)
        self.assertEqual(out[0, -1], 0)
        self.assertEqual(out[1, 0], 4)


def _decode_row_bounds(seq_lens, next_n):
    """Mirror of the window expansion done by ``top_k_per_row_decode``.

    ``seq_lens`` holds one length per *request* (shape ``(B,)`` or ``(B, 1)``),
    matching upstream ``IndexerTopK._row_ends()``. Each request expands into
    ``next_n`` speculative rows whose windows end at ``seq_len - next_n + 1 ...
    seq_len`` and all start at column 0, so the batch is *not* divided by
    ``next_n`` -- the element count and the row count differ by that factor.
    """
    seq_lens = np.asarray(seq_lens, dtype=np.int64).reshape(-1)
    batch = seq_lens.size
    offsets = np.arange(next_n, dtype=np.int64)
    ends = seq_lens.reshape(batch, 1) - next_n + 1 + offsets.reshape(1, next_n)
    starts = np.zeros(batch * next_n, dtype=np.int32)
    return starts, ends.clip(min=0).reshape(-1).astype(np.int32)


class TestContract(unittest.TestCase):
    def test_decode_row_ends_expand_per_request(self):
        """One length per request, ``next_n`` rows per request (not ``numel()/next_n``)."""
        s, e = _decode_row_bounds([10, 11], 3)
        np.testing.assert_array_equal(s, np.zeros(6, dtype=np.int32))
        np.testing.assert_array_equal(e, [8, 9, 10, 9, 10, 11])

        # Requests shorter than the speculative window clamp at 0, never negative.
        _, short_ends = _decode_row_bounds([2], 4)
        np.testing.assert_array_equal(short_ends, [0, 0, 1, 2])

    def test_short_window_pads_minus_one(self):
        logits = np.arange(24, dtype=np.float32).reshape(3, 8)
        s = np.array([0, 2, 5], dtype=np.int32)
        e = np.array([3, 4, 5], dtype=np.int32)  # lengths 3, 2, 0
        out = top_k_per_row_numpy(logits, s, e, 4)
        np.testing.assert_array_equal(out[0], [2, 1, 0, -1])
        np.testing.assert_array_equal(out[1], [3, 2, -1, -1])
        np.testing.assert_array_equal(out[2], [-1, -1, -1, -1])
        self.assertEqual(check_invariants(logits, s, e, 4, out), [])

    def test_window_clamped(self):
        logits = np.arange(10, dtype=np.float32).reshape(1, 10)
        s = np.array([-5], dtype=np.int32)
        e = np.array([99], dtype=np.int32)
        out = top_k_per_row_numpy(logits, s, e, 3)
        np.testing.assert_array_equal(out[0], [9, 8, 7])

    def test_full_width_selection(self):
        logits = np.arange(64, dtype=np.float32).reshape(1, 64)
        s = np.zeros(1, dtype=np.int32)
        e = np.full(1, 64, dtype=np.int32)
        out = top_k_per_row_numpy(logits, s, e, 64)
        np.testing.assert_array_equal(out[0], np.arange(63, -1, -1))

    def test_bf16_round_trip_values(self):
        """Values are compared in fp32 after a bf16 cast, as the NPU does."""
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(4, 512)).astype(np.float32)
        bf16 = (logits.view(np.uint32) & 0xFFFF0000).view(np.float32)
        s = np.zeros(4, dtype=np.int32)
        e = np.full(4, 512, dtype=np.int32)
        a = top_k_per_row_numpy(bf16, s, e, 128)
        self.assertEqual(check_invariants(bf16, s, e, 128, a), [])

    def test_many_rows_one_column(self):
        logits = np.full((17, 1), 2.5, dtype=np.float32)
        s = np.zeros(17, dtype=np.int32)
        e = np.ones(17, dtype=np.int32)
        out = top_k_per_row_numpy(logits, s, e, 5)
        self.assertTrue(np.all(out[:, 0] == 0))
        self.assertTrue(np.all(out[:, 1:] == -1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
