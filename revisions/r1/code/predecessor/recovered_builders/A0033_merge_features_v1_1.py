#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_features_v1_1.py
======================
v1.0 (126 维 + 3 fraud 列) + RPT_Operation 7 维 = v1.1 (133 维 + 3 fraud 列).

合并步骤:
  1. 加载 features_v1_0.parquet
  2. 加载 rpt_operation_features_v1_1.parquet
  3. 左 join (主键 firm_id × year)
  4. RPT 列缺失 fillna(0) (无关联交易记录 = 真实 0)
  5. 保持 fraud 列在最末 (训练脚本约定)

输出:
  data/processed/features/features_v1_1.parquet
  data/interim/v1_integration_stats/merge_features_v1_1_stats.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _v1_common import (
    get_logger,
    get_paths,
    log_mem,
    write_stats_json,
)


def merge_v1_1(
    v1_0_parquet: Path,
    rpt_parquet: Path,
    out_parquet: Path,
    out_stats: Path,
    log,
):
    log.info("=" * 60)
    log.info("Step 1: 加载 v1.0")
    log.info("=" * 60)
    v10 = pd.read_parquet(v1_0_parquet)
    log.info(f"v1.0 shape: {v10.shape}")
    v10["firm_id"] = v10["firm_id"].astype(str).str.zfill(6)
    v10["year"] = v10["year"].astype(int)
    log_mem(log, "after_load_v10")

    log.info("=" * 60)
    log.info("Step 2: 加载 RPT_Operation 7 维特征")
    log.info("=" * 60)
    rpt = pd.read_parquet(rpt_parquet)
    rpt["firm_id"] = rpt["firm_id"].astype(str).str.zfill(6)
    rpt["year"] = rpt["year"].astype(int)
    rpt_cols = [c for c in rpt.columns if c not in ["firm_id", "year"]]
    log.info(f"RPT shape: {rpt.shape}, cols: {rpt_cols}")

    log.info("=" * 60)
    log.info("Step 3: 左 join")
    log.info("=" * 60)

    # 把 fraud 列暂时移到末尾以便最终再放回
    fraud_cols_present = [c for c in v10.columns if c.startswith("fraud_")]
    feature_part = v10.drop(columns=fraud_cols_present)

    out = feature_part.merge(rpt, on=["firm_id", "year"], how="left")
    # RPT 缺失 fillna 0
    for c in rpt_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # 还原 fraud 列到末尾
    if fraud_cols_present:
        fraud_part = v10[["firm_id", "year"] + fraud_cols_present]
        out = out.merge(fraud_part, on=["firm_id", "year"], how="left")
        for c in fraud_cols_present:
            out[c] = out[c].fillna(0).astype("int8")

    log.info(f"v1.1 shape: {out.shape}")
    n_features = len(out.columns) - 2 - len(fraud_cols_present)
    log.info(f"特征维数 (不含 id/fraud): {n_features}")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False, compression="snappy")
    log.info(f"v1.1 写入: {out_parquet}")

    # 切分统计
    splits = {}
    for split_name, year_range in [("train", (2010, 2018)), ("val", (2019, 2020)), ("test", (2021, 2024))]:
        m = (out["year"] >= year_range[0]) & (out["year"] <= year_range[1])
        sub = out.loc[m]
        for fraud_col in fraud_cols_present:
            splits[f"{split_name}_{fraud_col}"] = {
                "n_total": int(len(sub)),
                "n_positive": int(sub[fraud_col].sum()),
                "positive_rate": float(sub[fraud_col].mean()) if len(sub) > 0 else 0.0,
            }

    stats = {
        "version": "v1.1",
        "input": {"v1_0": str(v1_0_parquet), "rpt_operation": str(rpt_parquet)},
        "output": str(out_parquet),
        "shape": list(out.shape),
        "n_total_features": n_features,
        "rpt_features": rpt_cols,
        "fraud_label_cols": fraud_cols_present,
        "main_label_recommendation": "fraud_v08_strict",
        "splits_by_design_A": splits,
        "notes": [
            "v1.1 是 Chapter 7 §7.4 异构图实验 + Chapter 6 多模态实验的最终特征集",
            "rpt_* 7 维与 KG E7 边互补: 特征捕捉数量, 边捕捉关系结构",
            "保留 v1.0 全部 126 维 + 3 fraud 列, 仅追加 7 维 RPT",
        ],
    }
    write_stats_json(stats, out_stats, log=log)


def main():
    p = argparse.ArgumentParser(description="合并 v1.0 + 7 维 RPT = v1.1")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--v1-0-parquet", type=Path, default=None)
    p.add_argument("--rpt-parquet", type=Path, default=None)
    p.add_argument("--out-parquet", type=Path, default=None)
    p.add_argument("--out-stats", type=Path, default=None)
    args = p.parse_args()

    paths = get_paths(args.project_root)
    log = get_logger("merge_features_v1_1", paths["stats"])

    v10 = args.v1_0_parquet or (paths["processed_features"] / "features_v1_0.parquet")
    rpt = args.rpt_parquet or (paths["processed_features"] / "rpt_operation_features_v1_1.parquet")
    out = args.out_parquet or (paths["processed_features"] / "features_v1_1.parquet")
    out_stats = args.out_stats or (paths["stats"] / "merge_features_v1_1_stats.json")

    log.info(f"v1.0: {v10}")
    log.info(f"RPT: {rpt}")
    log.info(f"out: {out}")

    merge_v1_1(v10, rpt, out, out_stats, log)
    log.info("✓ merge_features_v1_1 完成")


if __name__ == "__main__":
    main()
