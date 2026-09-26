#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_features_v1_0.py
======================
合并 v0.7 (109 维 + fraud) + audit (5) + pledge (6) + controller (6+1) = v1.0 (126+1 维 + 3 列 fraud).

合并步骤:
  1. 加载 v0.7 features_v0_7.parquet (51,675 行 × ~110 列)
  2. drop v0.7 自带 fraud 列 (改用 v0.8 三列标签)
  3. 左 join audit_features_v1_0.parquet (5 维)
  4. 左 join pledge_features_v1_0.parquet (6 维)
  5. 左 join controller_features_v1_0.parquet (7 维: 6 主 + 1 missingness)
  6. 左 join fraud_labels_v0_8.parquet (3 列: fraud_v07/strict/loose)
  7. fillna(0) for 缺失数值列, 输出 features_v1_0.parquet

输出:
  data/processed/features/features_v1_0.parquet
  data/interim/v1_integration_stats/merge_features_v1_0_stats.json

设计要点:
  - 保留 v0.7 全部 109 维原始特征列 (向后兼容)
  - 新增 18 维 (5+6+7) 排在尾部, 列名前缀清晰 (audit_/pld_/ctrl_) 便于 Chapter 6 模态切分实验
  - 主标签 = fraud_v08_strict, 训练时用此列
  - fraud_v07/loose 保留供 Chapter 7 §7.5 鲁棒性分析
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


# ============================================================
def merge_v1_0(
    v07_parquet: Path,
    audit_parquet: Path,
    pledge_parquet: Path,
    controller_parquet: Path,
    fraud_v08_parquet: Path,
    out_parquet: Path,
    out_stats: Path,
    log,
):
    log.info("=" * 60)
    log.info("Step 1: 加载 v0.7 features")
    log.info("=" * 60)
    v07 = pd.read_parquet(v07_parquet)
    log.info(f"v0.7 shape: {v07.shape}")
    log.info(f"v0.7 列前 10: {v07.columns[:10].tolist()}, ... , 末 5: {v07.columns[-5:].tolist()}")

    # 规整 firm_id
    v07["firm_id"] = v07["firm_id"].astype(str).str.zfill(6)
    v07["year"] = v07["year"].astype(int)
    log_mem(log, "after_load_v07")

    # drop fraud 列 (改用 v0.8 三列)
    fraud_cols_in_v07 = [c for c in v07.columns if c.startswith("fraud") or c.startswith("label")]
    if fraud_cols_in_v07:
        log.info(f"v0.7 中已有 fraud-like 列: {fraud_cols_in_v07}, drop 之后用 v0.8")
        v07 = v07.drop(columns=fraud_cols_in_v07)

    # ---------- step 2: load audit ----------
    log.info("=" * 60)
    log.info("Step 2: load audit_features_v1_0")
    log.info("=" * 60)
    audit = pd.read_parquet(audit_parquet)
    audit["firm_id"] = audit["firm_id"].astype(str).str.zfill(6)
    audit["year"] = audit["year"].astype(int)
    log.info(f"audit shape: {audit.shape}, cols: {audit.columns.tolist()}")

    # ---------- step 3: load pledge ----------
    log.info("=" * 60)
    log.info("Step 3: load pledge_features_v1_0")
    log.info("=" * 60)
    pledge = pd.read_parquet(pledge_parquet)
    pledge["firm_id"] = pledge["firm_id"].astype(str).str.zfill(6)
    pledge["year"] = pledge["year"].astype(int)
    log.info(f"pledge shape: {pledge.shape}, cols: {pledge.columns.tolist()}")

    # ---------- step 4: load controller ----------
    log.info("=" * 60)
    log.info("Step 4: load controller_features_v1_0")
    log.info("=" * 60)
    controller = pd.read_parquet(controller_parquet)
    controller["firm_id"] = controller["firm_id"].astype(str).str.zfill(6)
    controller["year"] = controller["year"].astype(int)
    log.info(f"controller shape: {controller.shape}, cols: {controller.columns.tolist()}")

    # ---------- step 5: 顺次左 join ----------
    log.info("=" * 60)
    log.info("Step 5: 顺次左 join (以 v0.7 firm-year 为主键)")
    log.info("=" * 60)

    out = v07.merge(audit, on=["firm_id", "year"], how="left")
    log.info(f"after audit join: {out.shape}")

    out = out.merge(pledge, on=["firm_id", "year"], how="left")
    log.info(f"after pledge join: {out.shape}")

    out = out.merge(controller, on=["firm_id", "year"], how="left")
    log.info(f"after controller join: {out.shape}")
    log_mem(log, "after_all_joins")

    # ---------- step 6: load fraud labels ----------
    log.info("=" * 60)
    log.info("Step 6: load fraud_labels_v0_8")
    log.info("=" * 60)
    fraud = pd.read_parquet(fraud_v08_parquet)
    fraud["firm_id"] = fraud["firm_id"].astype(str).str.zfill(6)
    fraud["year"] = fraud["year"].astype(int)
    log.info(f"fraud shape: {fraud.shape}, cols: {fraud.columns.tolist()}")

    out = out.merge(fraud, on=["firm_id", "year"], how="left")
    # 三列 fraud 缺失填 0 (即 v0.7 全集中部分 firm-year 不在 fraud_v0_8 里 = 都 0)
    for c in ["fraud_v07", "fraud_v08_strict", "fraud_v08_loose"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype("int8")

    log.info(f"after fraud join: {out.shape}")

    # ---------- step 7: 缺失值填充 (新增列若有 NaN, 归零 → 与 align 阶段重复但保证幂等) ----------
    log.info("=" * 60)
    log.info("Step 7: 数值列缺失 fillna(0)")
    log.info("=" * 60)

    new_feature_cols = (
        [c for c in audit.columns if c not in ["firm_id", "year"]]
        + [c for c in pledge.columns if c not in ["firm_id", "year"]]
        + [c for c in controller.columns if c not in ["firm_id", "year"]]
    )

    for c in new_feature_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # ---------- step 8: 列序整理 ----------
    log.info("=" * 60)
    log.info("Step 8: 列序整理")
    log.info("=" * 60)

    id_cols = ["firm_id", "year"]
    fraud_cols = ["fraud_v07", "fraud_v08_strict", "fraud_v08_loose"]
    fraud_cols = [c for c in fraud_cols if c in out.columns]
    audit_cols_ordered = [c for c in audit.columns if c not in ["firm_id", "year"]]
    pledge_cols_ordered = [c for c in pledge.columns if c not in ["firm_id", "year"]]
    controller_cols_ordered = [c for c in controller.columns if c not in ["firm_id", "year"]]

    new_cols_set = set(audit_cols_ordered + pledge_cols_ordered + controller_cols_ordered + fraud_cols)
    v07_features = [c for c in v07.columns if c not in id_cols and c not in new_cols_set]

    final_order = (
        id_cols
        + v07_features
        + audit_cols_ordered
        + pledge_cols_ordered
        + controller_cols_ordered
        + fraud_cols
    )
    final_order = [c for c in final_order if c in out.columns]
    out = out[final_order]

    log.info(f"最终列数: {len(final_order)}")
    log.info(f"  id: {len(id_cols)} ({id_cols})")
    log.info(f"  v0.7 features: {len(v07_features)}")
    log.info(f"  audit: {len(audit_cols_ordered)} {audit_cols_ordered}")
    log.info(f"  pledge: {len(pledge_cols_ordered)} {pledge_cols_ordered}")
    log.info(f"  controller: {len(controller_cols_ordered)} {controller_cols_ordered}")
    log.info(f"  fraud labels: {len(fraud_cols)} {fraud_cols}")
    n_features = len(v07_features) + len(audit_cols_ordered) + len(pledge_cols_ordered) + len(controller_cols_ordered)
    log.info(f"特征总维数 (不含 id/fraud): {n_features}")

    # ---------- step 9: 持久化 ----------
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False, compression="snappy")
    log.info(f"v1.0 写入: {out_parquet}  ({len(out)} 行 × {len(final_order)} 列)")
    log_mem(log, "after_write")

    # ---------- step 10: 切分统计 (与 v0.7 论文设计 A 一致) ----------
    splits = {}
    for split_name, year_range in [("train", (2010, 2018)), ("val", (2019, 2020)), ("test", (2021, 2024))]:
        m = (out["year"] >= year_range[0]) & (out["year"] <= year_range[1])
        sub = out.loc[m]
        for fraud_col in fraud_cols:
            splits[f"{split_name}_{fraud_col}"] = {
                "n_total": int(len(sub)),
                "n_positive": int(sub[fraud_col].sum()),
                "positive_rate": float(sub[fraud_col].mean()) if len(sub) > 0 else 0.0,
            }

    stats = {
        "version": "v1.0",
        "input": {
            "v07": str(v07_parquet),
            "audit": str(audit_parquet),
            "pledge": str(pledge_parquet),
            "controller": str(controller_parquet),
            "fraud_v08": str(fraud_v08_parquet),
        },
        "output": str(out_parquet),
        "shape": list(out.shape),
        "n_total_features": n_features,
        "v07_features": len(v07_features),
        "new_features": {
            "audit": audit_cols_ordered,
            "pledge": pledge_cols_ordered,
            "controller": controller_cols_ordered,
        },
        "fraud_label_cols": fraud_cols,
        "main_label_recommendation": "fraud_v08_strict",
        "splits_by_design_A": splits,
        "notes": [
            "v1.0 是 Chapter 6/7 后续实验的主特征集",
            "训练用 fraud_v08_strict (主标签), Chapter 7 §7.5 切换 fraud_v07/loose 做敏感性分析",
            "所有新增列前缀清晰: audit_/pld_/ctrl_, 便于按模态分组实验",
            "ctrl_data_missing_flag 是 bonus 维度, 单列 missingness 信号",
        ],
    }
    write_stats_json(stats, out_stats, log=log)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="合并 v0.7 + 17/18 维 = v1.0")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--v07-parquet", type=Path, default=None)
    p.add_argument("--audit-parquet", type=Path, default=None)
    p.add_argument("--pledge-parquet", type=Path, default=None)
    p.add_argument("--controller-parquet", type=Path, default=None)
    p.add_argument("--fraud-v08-parquet", type=Path, default=None)
    p.add_argument("--out-parquet", type=Path, default=None)
    p.add_argument("--out-stats", type=Path, default=None)
    args = p.parse_args()

    paths = get_paths(args.project_root)
    log = get_logger("merge_features_v1_0", paths["stats"])

    v07 = args.v07_parquet or (paths["processed_features"] / "features_v0_7.parquet")
    audit = args.audit_parquet or (paths["processed_features"] / "audit_features_v1_0.parquet")
    pledge = args.pledge_parquet or (paths["processed_features"] / "pledge_features_v1_0.parquet")
    ctrl = args.controller_parquet or (paths["processed_features"] / "controller_features_v1_0.parquet")
    fraud = args.fraud_v08_parquet or (paths["processed_labels"] / "fraud_labels_v0_8.parquet")
    out = args.out_parquet or (paths["processed_features"] / "features_v1_0.parquet")
    out_stats = args.out_stats or (paths["stats"] / "merge_features_v1_0_stats.json")

    log.info(f"v07: {v07}")
    log.info(f"audit: {audit}")
    log.info(f"pledge: {pledge}")
    log.info(f"controller: {ctrl}")
    log.info(f"fraud: {fraud}")
    log.info(f"out: {out}")

    merge_v1_0(v07, audit, pledge, ctrl, fraud, out, out_stats, log)
    log.info("✓ merge_features_v1_0 完成")


if __name__ == "__main__":
    main()
