"""
prepare_node_features_v1_1.py
==============================

Convert features_v1_1.parquet (51,675 × 139, firm-year × {financial, audit, pledge, control, RPT} features + 3 fraud labels)
to the GNN training KG node-feature format node_features_v1_1.parquet.

Design conventions:
  - Column structure matches node_features_v0_8.parquet (first 9 columns are metadata, followed by fraud labels then features)
  - Expands fraud columns from 1 to 3 (fraud_v07 / fraud_v08_strict / fraud_v08_loose)
  - Node id (node_id) follows v0_8 KG's "C:000001.SZ" format
  - join key: firm_id (6 -digit, "000001") ↔ ts_code prefix ("000001.SZ".split('.')[0])

Input:
  data/processed/features/features_v1_1.parquet     ((new, main feature matrix))
  data/processed/kg/node_features_v0_8.parquet      (legacy, source of node_id/ts_code metadata)

Output:
  data/processed/kg/node_features_v1_1.parquet      (51675 × ~140)
  data/interim/v1_integration_stats/prepare_node_features_v1_1_stats.json

Usage:
  uv run --project ~/code python scripts/p7_gnn/prepare_node_features_v1_1.py
  uv run --project ~/code python scripts/p7_gnn/prepare_node_features_v1_1.py \\
      --project-root /path/to/project
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Column-schema contract (output schema, strict order)
# ============================================================================

METADATA_COLS = [
    "node_id",       # "C:000001.SZ"
    "ts_code",       # "000001.SZ"
    "firm_id",       # "000001"  (added for downstream joins)
    "year",          # int 2010-2024
    "list_date",
    "delist_date",
    "industry",
    "market",
    "name",
]
FRAUD_COLS = [
    "fraud_v07",
    "fraud_v08_strict",   # ★ primary label
    "fraud_v08_loose",
]
FEATURE_PREFIXES = (
    "feat_",     # insider 17
    "fin_",      # 18 financial derived features (M-score / Z-score / DSRI etc.)
    "fini_",     # 69 tushare financial ratios
    "audit_",    # 5 (NEW)
    "pld_",      # 6 (NEW)
    "ctrl_",     # 7 (new, includes ctrl_data_missing_flag)
    "rpt_",      # 7 (NEW)
)
EXPECTED_FEATURE_COUNT = 17 + 18 + 69 + 5 + 6 + 7 + 7  # = 129


# ============================================================================
def setup_logger(name: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{name}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ============================================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--features-v11", type=Path, default=None,
                   help="features_v1_1.parquet path")
    p.add_argument("--node-features-v08", type=Path, default=None,
                   help="node_features_v0_8.parquet path (source of node_id/ts_code metadata)")
    p.add_argument("--out-parquet", type=Path, default=None)
    p.add_argument("--out-stats", type=Path, default=None)
    args = p.parse_args()

    proj = args.project_root or (Path.home() / "code" / "foa_project")
    if not proj.exists():
        print(f"project root directory does not exist: {proj}", file=sys.stderr)
        return 1

    feat_v11 = args.features_v11 or (
        proj / "data" / "processed" / "features" / "features_v1_1.parquet")
    nf_v08 = args.node_features_v08 or (
        proj / "data" / "processed" / "kg" / "node_features_v0_8.parquet")
    out_parq = args.out_parquet or (
        proj / "data" / "processed" / "kg" / "node_features_v1_1.parquet")
    out_stats = args.out_stats or (
        proj / "data" / "interim" / "v1_integration_stats"
        / "prepare_node_features_v1_1_stats.json")
    log_dir = proj / "data" / "interim" / "v1_integration_stats"

    log = setup_logger("prepare_node_features_v1_1", log_dir)
    log.info("=" * 60)
    log.info("KG node-feature preparation (v0.7 → v1.1)")
    log.info("=" * 60)
    log.info("PROJECT_ROOT       : %s", proj)
    log.info("features_v1_1      : %s", feat_v11)
    log.info("node_features_v0_8 : %s", nf_v08)
    log.info("out_parquet        : %s", out_parq)

    if not feat_v11.exists():
        log.error("features_v1_1.parquet does not exist: %s", feat_v11)
        return 2
    if not nf_v08.exists():
        log.error("node_features_v0_8.parquet does not exist: %s", nf_v08)
        return 2

    # ---------- step 1: reading features_v1_1 ----------
    log.info("Step 1: reading features_v1_1.parquet")
    v11 = pd.read_parquet(feat_v11)
    log.info("  shape: %s", v11.shape)
    v11["firm_id"] = v11["firm_id"].astype(str).str.zfill(6)
    v11["year"] = v11["year"].astype(int)

    # ---------- step 2: Extract node identity from node_features_v0_8 ----------
    log.info("Step 2: Extract node_id / ts_code metadata (from v0_8 KG)")
    # v1.1 already has list_date/delist_date/industry/market/name; pull only node_id+ts_code from v0_8
    v08_meta_cols = ["node_id", "ts_code", "year"]
    v08 = pd.read_parquet(nf_v08, columns=v08_meta_cols)
    v08["firm_id"] = v08["ts_code"].astype(str).str.split(".").str[0].str.zfill(6)
    v08["year"] = v08["year"].astype(int)
    log.info("  v0_8 metadata row count: %d", len(v08))

    # ---------- step 3: primary-key join (firm_id, year) ----------
    log.info("Step 3: primary-key join (firm_id, year)")
    metadata_payload = v08[["firm_id", "year", "node_id", "ts_code"]].drop_duplicates(
        ["firm_id", "year"])
    merged = v11.merge(metadata_payload, on=["firm_id", "year"], how="left")
    n_unmatched = int(merged["node_id"].isna().sum())
    log.info("  v1.1 row count: %d", len(merged))
    log.info("  unmatched (no node_id): %d", n_unmatched)
    if n_unmatched > 0:
        log.warning("  ⚠ %d v1.1 firm-year rows have no node_id match in v0_8 KG; these rows will be dropped",
                    n_unmatched)
        sample_unmatched = merged.loc[merged["node_id"].isna(),
                                       ["firm_id", "year"]].head(5)
        log.warning("  example: %s", sample_unmatched.values.tolist())
        merged = merged.dropna(subset=["node_id"]).copy()
        log.info("  kept %d rows", len(merged))

    # ---------- step 4: tidy column order ----------
    log.info("Step 4: tidy column order (metadata → fraud → feature)")
    fraud_in_df = [c for c in FRAUD_COLS if c in merged.columns]
    if len(fraud_in_df) != 3:
        log.error("Missing fraud columns: expected %s, got %s", FRAUD_COLS, fraud_in_df)
        return 3

    # Collect all feature columns (by prefix order)
    feature_cols_ordered = []
    for prefix in FEATURE_PREFIXES:
        prefix_cols = [c for c in merged.columns if c.startswith(prefix)]
        prefix_cols = sorted(prefix_cols)
        feature_cols_ordered.extend(prefix_cols)
        log.info("  %s : %d columns", prefix, len(prefix_cols))
    log.info("  total feature columns: %d (expected %d)",
             len(feature_cols_ordered), EXPECTED_FEATURE_COUNT)
    if len(feature_cols_ordered) != EXPECTED_FEATURE_COUNT:
        log.warning("  ⚠ feature-column count mismatch — check FEATURE_PREFIXES against actual data")

    # final column order
    final_cols = METADATA_COLS + FRAUD_COLS + feature_cols_ordered
    missing = [c for c in final_cols if c not in merged.columns]
    if missing:
        log.error("missingOutputcolumns: %s", missing)
        return 4
    out_df = merged[final_cols].copy()

    # ---------- step 5: dtype normalization ----------
    log.info("Step 5: dtype normalization")
    out_df["year"] = out_df["year"].astype("int16")
    for c in FRAUD_COLS:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(0).astype("int8")
    for c in feature_cols_ordered:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce").astype("float32")
    # Keep string columns as str
    for c in ["node_id", "ts_code", "firm_id", "list_date",
              "delist_date", "industry", "market", "name"]:
        out_df[c] = out_df[c].astype(str)

    # ---------- step 6: persist ----------
    out_parq.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parq, index=False, compression="snappy")
    log.info("✓ wrote: %s  (%d rows × %d columns, %.1f MB)",
             out_parq, len(out_df), out_df.shape[1],
             out_parq.stat().st_size / 1e6)

    # ---------- step 7: split-size sanity check ----------
    log.info("temporal split sanity:")
    splits = [("train", 2010, 2018), ("val", 2019, 2020), ("test", 2021, 2024)]
    splits_stats = {}
    for name, lo, hi in splits:
        m = out_df["year"].between(lo, hi)
        sub = out_df[m]
        s = {"n": int(len(sub))}
        for col in FRAUD_COLS:
            n_pos = int(sub[col].sum())
            s[f"{col}_pos"] = n_pos
            s[f"{col}_rate"] = round(n_pos / max(len(sub), 1), 4)
        splits_stats[name] = s
        log.info(
            "  %s %d-%d: %d rows | strict %d (%.2f%%) | v07 %d (%.2f%%) | loose %d (%.2f%%)",
            name, lo, hi, len(sub),
            s["fraud_v08_strict_pos"], s["fraud_v08_strict_rate"] * 100,
            s["fraud_v07_pos"], s["fraud_v07_rate"] * 100,
            s["fraud_v08_loose_pos"], s["fraud_v08_loose_rate"] * 100,
        )

    # ---------- step 8: write stats JSON ----------
    stats = {
        "version": "v1.1",
        "input": {
            "features_v11": str(feat_v11),
            "node_features_v08": str(nf_v08),
        },
        "output": {
            "parquet": str(out_parq),
            "n_rows": int(len(out_df)),
            "n_cols": int(out_df.shape[1]),
            "size_mb": round(out_parq.stat().st_size / 1e6, 1),
        },
        "schema": {
            "metadata_cols": METADATA_COLS,
            "fraud_cols": FRAUD_COLS,
            "feature_prefixes_and_counts": {
                p: len([c for c in feature_cols_ordered if c.startswith(p)])
                for p in FEATURE_PREFIXES
            },
            "n_features_total": len(feature_cols_ordered),
        },
        "modal_definitions_for_chapter7": {
            "M1_insider": ["feat_"],
            "M2_finance": ["fin_", "fini_"],
            "M3_insider_fin_core": ["feat_", "fin_"],
            "M4_insider_fini": ["feat_", "fini_"],
            "M5_v07_full": ["feat_", "fin_", "fini_"],
            "M6_M5_audit": ["feat_", "fin_", "fini_", "audit_"],
            "M7_M5_pledge": ["feat_", "fin_", "fini_", "pld_"],
            "M8_M5_controller": ["feat_", "fin_", "fini_", "ctrl_"],
            "M9_M5_rpt": ["feat_", "fin_", "fini_", "rpt_"],
            "M10_v10_full": ["feat_", "fin_", "fini_", "audit_", "pld_", "ctrl_"],
            "M11_v11_full": ["feat_", "fin_", "fini_", "audit_", "pld_",
                              "ctrl_", "rpt_"],
        },
        "splits_design_A": splits_stats,
        "n_unmatched_dropped": n_unmatched,
        "_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
    log.info("✓ stats JSON: %s", out_stats)

    log.info("=" * 60)
    log.info("✓ prepare_node_features_v1_1 done")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
