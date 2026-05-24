"""
gnn_baseline_common.py
========================

Shared utilities for the GNN baseline experiments reported in the paper.

Design conventions (consistent across all training scripts in this directory):
  - Temporal train/val/test split: 2010-2018 train / 2019-2020 val / 2021-2024 test
  - Evaluation metrics: AUC-ROC, AP, Precision@5%, Recall@10%FPR, F1@best-threshold
  - Default global random seed: 42 (overridable via the --seed CLI flag in each script)
  - Pairwise AUC significance test: DeLong's paired test (Sun & Xu 2014 fast variant)

Project root resolution:
  - If the environment variable PROJECT_ROOT is set, use it.
  - Otherwise default to ~/code/foa_project. This default exists only so that
    each script runs out-of-the-box on the original training machine; on any other
    setup, set PROJECT_ROOT to the directory that contains data/processed/kg/.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Project root: PROJECT_ROOT environment variable preferred; otherwise fall back to
# the original training machine's path (override on any other host).
PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT", str(Path.home() / "code" / "foa_project")))
KG_DIR = PROJECT_ROOT / "data" / "processed" / "kg"
RESULTS_DIR = PROJECT_ROOT / "data" / "interim" / "p7_gnn_results"
LOG_DIR = PROJECT_ROOT / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Temporal split boundaries
SPLIT_TRAIN_YEARS = list(range(2010, 2019))   # 2010-2018
SPLIT_VAL_YEARS = [2019, 2020]                # 2019-2020
SPLIT_TEST_YEARS = list(range(2021, 2025))    # 2021-2024


# ============================================================================
# Data loading
# ============================================================================

def load_node_features(features_path: Path = None) -> pd.DataFrame:
    """Load node_features_v0_8.parquet (51,675 rows x 113 columns)."""
    if features_path is None:
        features_path = KG_DIR / "node_features_v0_8.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"node_features_v0_8.parquet not found: {features_path}")
    return pd.read_parquet(features_path)


def load_edge_index(edge_path: Path = None) -> pd.DataFrame:
    """Load global_edge_index.parquet (4M+ integer-indexed edges)."""
    if edge_path is None:
        edge_path = KG_DIR / "global_edge_index.parquet"
    if not edge_path.exists():
        raise FileNotFoundError(f"global_edge_index.parquet not found: {edge_path}")
    return pd.read_parquet(edge_path)


def load_node_index(idx_path: Path = None) -> pd.DataFrame:
    """Load node_id_index.parquet (node id <-> integer idx mapping)."""
    if idx_path is None:
        idx_path = KG_DIR / "node_id_index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(f"node_id_index.parquet not found: {idx_path}")
    return pd.read_parquet(idx_path)


# ============================================================================
# Temporal split
# ============================================================================

def temporal_split(features_df: pd.DataFrame) -> dict:
    """Return {train, val, test} DataFrames partitioned by year."""
    train_df = features_df[features_df["year"].isin(SPLIT_TRAIN_YEARS)].copy()
    val_df = features_df[features_df["year"].isin(SPLIT_VAL_YEARS)].copy()
    test_df = features_df[features_df["year"].isin(SPLIT_TEST_YEARS)].copy()
    return {"train": train_df, "val": val_df, "test": test_df}


# ============================================================================
# Evaluation metrics
# ============================================================================

def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute the five headline metrics used throughout the paper:
    AUC-ROC, AP, Precision@5%, Recall@10%FPR, F1@best-threshold.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve, roc_curve, f1_score,
    )
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Precision@5%
    n_top = max(1, int(0.05 * len(y_score)))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    p_at_5pct = float(y_true[top_idx].mean())

    # Recall@10%FPR
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= 0.10
    r_at_10fpr = float(tpr[mask][-1]) if mask.any() else 0.0

    # F1 at the threshold that maximises F1
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    f1_best = float(f1s.max())

    return {
        "auc": float(auc), "ap": float(ap),
        "p_at_5pct": p_at_5pct,
        "r_at_10fpr": r_at_10fpr,
        "f1_best": f1_best,
    }


# ============================================================================
# DeLong paired AUC test (Sun & Xu 2014 fast variant)
# ============================================================================

def delong_test(y_true: np.ndarray,
                 y_score_a: np.ndarray, y_score_b: np.ndarray) -> dict:
    """DeLong's paired AUC test. Returns dict with auc_a, auc_b, delta, z, p."""
    from scipy import stats
    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return {"auc_a": 0, "auc_b": 0, "delta": 0, "z": 0, "p": 1.0}

    def midrank(x):
        order = np.argsort(x)
        x_sorted = x[order]
        ranks = np.zeros_like(order, dtype=float)
        i = 0
        while i < len(x_sorted):
            j = i
            while j < len(x_sorted) and x_sorted[j] == x_sorted[i]:
                j += 1
            ranks[order[i:j]] = (i + j - 1) / 2 + 1
            i = j
        return ranks

    def auc_var(score):
        all_r = midrank(score)
        pos_r = midrank(score[pos])
        neg_r = midrank(score[neg])
        auc = (all_r[pos].sum() / n_pos - (n_pos + 1) / 2) / n_neg
        v01 = (all_r[pos] - pos_r) / n_neg
        v10 = 1 - (all_r[neg] - neg_r) / n_pos
        return auc, v01, v10

    auc_a, v01_a, v10_a = auc_var(np.asarray(y_score_a))
    auc_b, v01_b, v10_b = auc_var(np.asarray(y_score_b))
    s01 = np.cov(np.vstack([v01_a, v01_b]))[0, 1]
    s10 = np.cov(np.vstack([v10_a, v10_b]))[0, 1]
    var = ((np.var(v01_a, ddof=1) + np.var(v01_b, ddof=1) - 2 * s01) / n_pos +
           (np.var(v10_a, ddof=1) + np.var(v10_b, ddof=1) - 2 * s10) / n_neg)
    if var <= 0:
        return {"auc_a": float(auc_a), "auc_b": float(auc_b),
                "delta": float(auc_b - auc_a), "z": 0.0, "p": 1.0}
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b),
            "delta": float(auc_b - auc_a), "z": float(z), "p": float(p)}


# ============================================================================
# Logging
# ============================================================================

def setup_logger(task: str):
    import logging
    logger = logging.getLogger(task)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOG_DIR / f"{task}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def write_results(task: str, payload: dict) -> Path:
    payload["_task"] = task
    payload["_timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = RESULTS_DIR / f"{task}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                              default=str),
                   encoding="utf-8")
    tmp.replace(out)
    return out


# ============================================================================
# Self-check
# ============================================================================

if __name__ == "__main__":
    logger = setup_logger("gnn_baseline_common_test")
    logger.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logger.info("KG_DIR exists: %s", KG_DIR.exists())
    logger.info("RANDOM_SEED: %d", RANDOM_SEED)
    logger.info("Time split: train=%s, val=%s, test=%s",
                f"{SPLIT_TRAIN_YEARS[0]}-{SPLIT_TRAIN_YEARS[-1]}",
                SPLIT_VAL_YEARS,
                f"{SPLIT_TEST_YEARS[0]}-{SPLIT_TEST_YEARS[-1]}")
    try:
        feats = load_node_features()
        logger.info("node_features_v0_8 loaded: %d rows x %d cols",
                    len(feats), len(feats.columns))
        edges = load_edge_index()
        logger.info("global_edge_index loaded: %d edges", len(edges))
        nodes = load_node_index()
        logger.info("node_id_index loaded: %d nodes", len(nodes))
        splits = temporal_split(feats)
        for k, v in splits.items():
            n_pos = (v["fraud"] == 1).sum() if "fraud" in v.columns else 0
            logger.info("  %s: %d rows, %d positives (%.2f%%)",
                        k, len(v), n_pos, n_pos / max(len(v), 1) * 100)
    except Exception as e:
        logger.error("Self-check failed: %s", e)
