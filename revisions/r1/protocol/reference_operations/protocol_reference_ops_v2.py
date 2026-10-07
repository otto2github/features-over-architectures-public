"""Pure reference operations for protocol specification. NOT a training runner.

No file discovery, network calls, model fitting or empirical dataset loading.
Inputs remain private in any future empirical use. Synthetic tests only were run
when this protocol was prepared.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import FrozenSet, Sequence
import numpy as np

@dataclass(frozen=True)
class LabeledRow:
    row_key: tuple[str, int]
    partition: str
    target: int | None
    document_keys: FrozenSet[tuple[str, str]]


def document_supervision_masks(rows: Sequence[LabeledRow]) -> tuple[np.ndarray, np.ndarray]:
    """Purge supervision by all direct usable keys; do not change labels.

    Document keys must already be generated with the pinned V26 normalizer.
    V/T are collected before either exclusion is applied. Negatives are kept.
    A caller must still verify source lineage, dates and the original cohort.
    """
    if len({r.row_key for r in rows}) != len(rows):
        raise ValueError('duplicate row key')
    if any(r.target not in (0, 1, None) for r in rows):
        raise ValueError('invalid target')
    if any(r.partition not in ('train', 'validation', 'test', 'outside') for r in rows):
        raise ValueError('invalid partition')
    v = {k for r in rows if r.partition == 'validation' and r.target == 1 for k in r.document_keys}
    t = {k for r in rows if r.partition == 'test' and r.target == 1 for k in r.document_keys}
    tr = np.array([r.partition == 'train' and r.target is not None and
                   not (r.target == 1 and bool(r.document_keys & (v | t))) for r in rows], dtype=bool)
    va = np.array([r.partition == 'validation' and r.target is not None and
                   not (r.target == 1 and bool(r.document_keys & t)) for r in rows], dtype=bool)
    return tr, va


def _uniform_donor_index(imputation_seed: int, feature_name: str,
                         row_key: tuple[str, int], pool_size: int) -> int:
    """Order-independent, unbiased discrete draw with deterministic PCG64."""
    if pool_size < 1:
        raise ValueError('empty donor pool')
    payload = json.dumps(['V27-B-v2', int(imputation_seed), str(feature_name),
                          str(row_key[0]), int(row_key[1])],
                         ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')
    rng = np.random.PCG64(seed)
    lim = (1 << 64) - ((1 << 64) % int(pool_size))
    while True:
        u = int(rng.random_raw())
        if u < lim:
            return u % int(pool_size)


def donor_complete_financial(
    matrix: np.ndarray, feature_names: Sequence[str],
    row_keys: Sequence[tuple[str, int]], partitions: Sequence[str],
    eligible: np.ndarray, financial_columns: Sequence[str], imputation_seed: int
) -> np.ndarray:
    """Replace only final nonfinite financial cells using training-only donors.

    Copies input; preserves all finite values and all nonfinancial columns.
    No labels, model seeds or scores are accepted. A missing eligible training
    donor is a hard error. The full 87-column membership guard belongs to the
    execution preflight; small column lists are allowed for synthetic tests.
    """
    x = np.asarray(matrix, dtype=np.float64)
    elig = np.asarray(eligible, dtype=bool)
    n, d = x.shape
    if (d != len(feature_names) or n != len(row_keys) or n != len(partitions)
            or elig.shape != (n,)):
        raise ValueError('shape mismatch')
    if len(set(feature_names)) != d or len(set(row_keys)) != n:
        raise ValueError('duplicate feature or row key')
    if not set(financial_columns) <= set(feature_names):
        raise ValueError('financial whitelist missing')
    if any(not c.startswith(('fin_', 'fini_')) for c in financial_columns):
        raise ValueError('nonfinancial column in financial whitelist')
    out = x.copy()
    tr = elig & (np.asarray(partitions) == 'train')
    canonical = sorted(range(n), key=lambda i: (str(row_keys[i][0]), int(row_keys[i][1])))
    for name in financial_columns:
        j = feature_names.index(name)
        donor_rows = [i for i in canonical if tr[i] and np.isfinite(x[i, j])]
        if not donor_rows:
            raise ValueError('NO_FINITE_ELIGIBLE_TRAINING_DONOR:' + name)
        for i in range(n):
            if elig[i] and not np.isfinite(x[i, j]):
                donor = donor_rows[_uniform_donor_index(imputation_seed, name, row_keys[i], len(donor_rows))]
                out[i, j] = x[donor, j]
    return out


def ranked_indices(scores: Sequence[float], row_keys: Sequence[tuple[str, int]]) -> list[int]:
    s = np.asarray(scores, dtype=float)
    if len(s) != len(row_keys) or not np.isfinite(s).all():
        raise ValueError('invalid scores')
    return sorted(range(len(s)), key=lambda i: (-s[i], str(row_keys[i][0]), int(row_keys[i][1])))


def full_budget_attribution(scores, row_keys, y, d_positive, k: int) -> dict[str, int]:
    """One full-population ranking, then D/O membership attribution."""
    yy = np.asarray(y, dtype=int)
    dd = np.asarray(d_positive, dtype=bool)
    if not (len(scores) == len(yy) == len(dd)) or not np.isin(yy, [0, 1]).all():
        raise ValueError('invalid vectors')
    if not (1 <= k <= len(yy)) or np.any(dd & (yy != 1)):
        raise ValueError('invalid budget or D positive mask')
    ix = ranked_indices(scores, row_keys)[:k]
    return dict(full=int(yy[ix].sum()), D=int((yy[ix] * dd[ix]).sum()),
                O=int((yy[ix] * (~dd[ix])).sum()))
