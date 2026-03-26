
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def top_svd_subspace(X: np.ndarray, rank: int) -> np.ndarray:
    X = X.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:rank].T

def fit_conflict_subspace_from_records(records: List[Dict[str, Any]], layer: int, rep_field: str='V_edit', slot: Optional[str]=None, pos_families=('true_override',), neg_families=('compatible_refinement', 'false_trigger'), rank: int=1):
    subset = [r for r in records if r['layer'] == layer and (slot is None or r['target_slot'] == slot) and (r['family'] in pos_families or r['family'] in neg_families)]
    if len(subset) < 8:
        return None
    X_pos = np.stack([r[rep_field] for r in subset if r['family'] in pos_families], axis=0).astype(np.float32)
    X_neg = np.stack([r[rep_field] for r in subset if r['family'] in neg_families], axis=0).astype(np.float32)
    if len(X_pos) < 4 or len(X_neg) < 4:
        return None
    mu_pos = X_pos.mean(axis=0, keepdims=True)
    mu_neg = X_neg.mean(axis=0, keepdims=True)
    mean_dir = (mu_pos - mu_neg).astype(np.float32)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-08)
    diff_mat = np.concatenate([X_pos - mu_neg, X_neg - mu_pos], axis=0)
    U = top_svd_subspace(diff_mat, rank=max(rank, 1))
    U[:, 0] = mean_dir.reshape(-1)
    q, _ = np.linalg.qr(U)
    U = q[:, :rank]
    return U.astype(np.float32)

def pairwise_subspace_similarity(U1: np.ndarray, U2: np.ndarray) -> float:
    M = U1.T @ U2
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(s))
