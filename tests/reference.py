"""Reference (CPU) semantics for TopKPerRow.

This module is the *executable specification* of the operator.  The Ascend C
kernel, the Triton reference kernel and the aclnn/torch bindings are all
expected to reproduce these exact semantics.

Operator contract
-----------------
    top_k_per_row(logits, row_starts, row_ends, topk) -> indices

    logits      : (num_rows, width) float32  - row-major, may be strided
    row_starts  : (num_rows,) int32          - inclusive window start
    row_ends    : (num_rows,) int32          - exclusive window end
    topk        : int                        - 1 <= topk <= TOPK_MAX (2048)
    indices     : (num_rows, topk) int32     - out

Per row ``r`` the operator selects the ``topk`` largest values inside the
column window ``[row_starts[r], row_ends[r])`` and writes their *column
indices* in descending value order.

Rules that the implementation must honour
-----------------------------------------
1. Window clamp: ``s = clamp(row_starts[r], 0, width)``,
   ``e = clamp(row_ends[r], 0, width)``, ``L = max(0, e - s)``.
2. ``L >= topk``  -> the row is fully filled with column indices from ``[s, e)``.
3. ``L <  topk``  -> the first ``L`` slots hold the sorted valid indices and
   the remaining ``topk - L`` slots are ``-1``.
4. Ties are broken by *ascending column index* (deterministic; the CUDA kernel
   makes no promise here, we do, so that tests are stable).
5. ``NaN`` ranks below ``-inf`` (it can never be selected while a finite
   candidate exists).  ``+/-0.0`` compare equal.
6. Output ordering is by value descending, never by index.

The reference is intentionally written twice (``numpy`` vectorised and a plain
python loop) so that the test-suite can cross-check the specification itself
before it is used as ground truth for device kernels.
"""

from __future__ import annotations

import numpy as np

TOPK_MAX = 2048


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


def top_k_per_row_numpy(
    logits: np.ndarray,
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    topk: int,
) -> np.ndarray:
    """Vectorised reference implementation (slow but obviously correct)."""
    logits = np.asarray(logits, dtype=np.float32)
    num_rows, width = logits.shape
    out = np.full((num_rows, topk), -1, dtype=np.int32)

    starts = np.clip(np.asarray(row_starts, dtype=np.int64), 0, width)
    ends = np.clip(np.asarray(row_ends, dtype=np.int64), 0, width)

    for r in range(num_rows):
        s, e = int(starts[r]), int(ends[r])
        if e <= s:
            continue
        row = logits[r, s:e].astype(np.float64)

        # NaN must lose against every non-NaN value (including -inf).
        nan_mask = np.isnan(row)
        key = np.where(nan_mask, -np.inf, row)
        # Sort by (-value, +index) so that ties resolve to the smaller index.
        idx = np.arange(e - s, dtype=np.int64)
        order = np.lexsort((idx, -key))
        # NaN block (if any) is placed at the very end, ordered by index.
        if nan_mask.any():
            valid = order[~nan_mask[order]]
            nans = order[nan_mask[order]]
            order = np.concatenate([valid, nans])

        take = min(topk, order.size)
        out[r, :take] = (order[:take] + s).astype(np.int32)
    return out


def top_k_per_row_loop(
    logits: np.ndarray,
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    topk: int,
) -> np.ndarray:
    """Plain-python reference, no numpy tricks. Used to cross-check above."""
    logits = np.asarray(logits, dtype=np.float32)
    num_rows, width = logits.shape
    out = np.full((num_rows, topk), -1, dtype=np.int32)

    for r in range(num_rows):
        s = _clamp(int(row_starts[r]), 0, width)
        e = _clamp(int(row_ends[r]), 0, width)
        cand = []
        for c in range(s, e):
            v = float(logits[r, c])
            if v != v:  # NaN
                rank = (1, 0.0, c)          # last bucket, value ignored
            else:
                rank = (0, -v, c)           # value desc, then index asc
            cand.append((rank, c))
        cand.sort()
        for i in range(min(topk, len(cand))):
            out[r, i] = cand[i][1]
    return out


# --------------------------------------------------------------------------
# helpers used by tests: semantic properties instead of a second algorithm
# --------------------------------------------------------------------------

def check_invariants(
    logits: np.ndarray,
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    topk: int,
    indices: np.ndarray,
) -> list[str]:
    """Return a list of human readable contract violations (empty == OK)."""
    logits = np.asarray(logits, dtype=np.float32)
    num_rows, width = logits.shape
    problems: list[str] = []
    if indices.shape != (num_rows, topk):
        return [f"shape {indices.shape} != {(num_rows, topk)}"]
    if indices.dtype != np.int32:
        problems.append(f"dtype {indices.dtype} != int32")

    for r in range(num_rows):
        s = _clamp(int(row_starts[r]), 0, width)
        e = _clamp(int(row_ends[r]), 0, width)
        L = max(0, e - s)
        row = indices[r]
        n_valid = int((row >= 0).sum())
        if n_valid != min(topk, L):
            problems.append(f"row {r}: {n_valid} valid slots, expected {min(topk, L)}")
        if len(set(int(x) for x in row[:n_valid])) != n_valid:
            problems.append(f"row {r}: duplicate indices")
        for x in row[:n_valid]:
            if not (s <= int(x) < e):
                problems.append(f"row {r}: index {x} outside window [{s},{e})")
                break
        if n_valid and not np.all(row[n_valid:] == -1):
            problems.append(f"row {r}: padding is not -1")
        # descending values, ties -> ascending index
        vals = [float(logits[r, int(x)]) for x in row[:n_valid]]
        for a in range(n_valid - 1):
            va, vb = vals[a], vals[a + 1]
            ia, ib = int(row[a]), int(row[a + 1])
            if va != va:  # NaN
                continue
            if vb == vb and va < vb:
                problems.append(f"row {r}: not descending at {a}")
                break
            if vb == vb and va == vb and ia > ib:
                problems.append(f"row {r}: tie not index-ascending at {a}")
                break
        # completeness: nothing outside the selection can beat the k-th value
        if 0 < n_valid == topk < L:
            kth = vals[-1]
            outside = logits[r, s:min(e, s + 1)]  # cheap probe, real check below
            windows = np.concatenate([logits[r, s:e]])
            sel = set(int(x) - s for x in row[:n_valid])
            rest = [windows[i] for i in range(L) if i not in sel]
            if rest:
                rest = np.asarray(rest, dtype=np.float32)
                rest = rest[~np.isnan(rest)]
                if rest.size and np.nanmax(rest) > kth:
                    problems.append(f"row {r}: missed a larger value")
    return problems
