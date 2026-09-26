#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_controller_features.py
================================
HLD_Contrshr 实控人 + 两权分离度 → 6 维实控人特征 + 1 维 missingness flag.

输出 7 维 (firm-year 聚合):
  ctrl_separation              : 两权分离度 % (S0704c - S0704b 或字段 Seperation), 论文金牌特征
  ctrl_ownership               : 实控人现金流权 % (S0704b)
  ctrl_voting                  : 实控人控制权 % (S0704c)
  ctrl_separation_p90_flag     : 二值 (Seperation > 17.45% 极端分离)
  ctrl_no_controller_flag      : 二值 (S0701a == 0 / 实控人姓名 NULL → 无实控人, 治理混乱信号)
  ctrl_state_owned_flag        : 二值 (S0702b 含国资关键字)
  ctrl_data_missing_flag       : ★ bonus 维度 (本 firm-year 在 HLD_Contrshr 中无记录, 缺失即信号)

学术依据:
  La Porta et al. (1999) "Corporate Ownership Around the World"
  Claessens, Djankov & Lang (2002) "Disentangling the Incentive and Entrenchment Effects"
  中国 A 股: 高分离度 → 隧道效应 (tunneling) → 财务造假, 显著正相关

数据特性 (依据 inventory):
  171,933 行 → A 股清洁 108,738 行 / 4,105 公司
  Seperation 非空率 92.0%, P50=0.29% / P90=17.45% / P99=28.33%
  firm-year 聚合后 40,366 行, 与 v0.7 全集 78% 覆盖, 缺失 22% 默认 fillna(0)

聚合策略:
  - 同一 firm × Reptdt-year 可能多条 (披露更新), 取该年最新一条 (按 Reptdt 取 max)
  - 极少数同年多条记录差异大: 取 ctrl_voting 最大者 (代表年末控制结构)

输出:
  data/processed/features/controller_features_v1_0.parquet
  data/interim/v1_integration_stats/integrate_controller_stats.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _v1_common import (
    extract_year,
    get_logger,
    get_paths,
    log_mem,
    read_csmar_xlsx,
    write_stats_json,
)

# 国资关键字 (实控人性质 S0702b 文本)
STATE_OWNED_KEYWORDS = [
    "国有", "国资", "国务院", "中央", "政府", "财政部", "国务院国资委",
    "央企", "省国资委", "市国资委", "县国资委",
    "中央国家机关", "地方政府", "事业单位", "国有独资",
]

P90_SEPERATION_THRESHOLD = 17.45  # inventory 给出的 P90 值 (%)


# ============================================================
def build_controller_features(
    hld_xlsx: Path,
    base_firm_year_parquet: Path,
    out_parquet: Path,
    out_stats: Path,
    log,
):
    # ---------- step 1: 读 HLD_Contrshr ----------
    log.info("=" * 60)
    log.info("Step 1: 读 HLD_Contrshr")
    log.info("=" * 60)

    df = read_csmar_xlsx(hld_xlsx, code_col="Stkcd", log=log)
    log.info(f"清洁后行数: {len(df)}")
    log.info(f"列: {df.columns.tolist()}")
    log_mem(log, "after_read_hld")

    # 识别字段名 (CSMAR 字段命名偶有变体)
    seperation_col = None
    for c in ["Seperation", "Separation", "separation"]:
        if c in df.columns:
            seperation_col = c
            break

    # ---------- step 2: 提取 year ----------
    df["year"] = extract_year(df["Reptdt"]).astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    log.info(f"提取 year 后: {len(df)} 行, year 范围 {df['year'].min()}-{df['year'].max()}")

    # ---------- step 3: 数值字段转换 ----------
    for col in ["S0704a", "S0704b", "S0704c"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if seperation_col is not None:
        df[seperation_col] = pd.to_numeric(df[seperation_col], errors="coerce")

    # 若缺 Seperation 列, 用 S0704c - S0704b 估算
    if seperation_col is None and "S0704c" in df.columns and "S0704b" in df.columns:
        df["_separation_calc"] = df["S0704c"] - df["S0704b"]
        seperation_col = "_separation_calc"
        log.info("Seperation 字段缺失, 用 S0704c - S0704b 估算")

    # ---------- step 4: 同 firm-year 多条记录: 取年末最新且控制权最大者 ----------
    log.info("=" * 60)
    log.info("Step 4: firm-year 聚合 (取年末最新)")
    log.info("=" * 60)

    df["_reptdt_dt"] = pd.to_datetime(df["Reptdt"], errors="coerce")
    df = df.sort_values(["Stkcd", "year", "_reptdt_dt", "S0704c"], ascending=[True, True, False, False])
    fy = df.drop_duplicates(["Stkcd", "year"], keep="first").copy()
    log.info(f"firm-year 去重后: {len(fy)} 行 / {fy['Stkcd'].nunique()} 公司")

    # ---------- step 5: 6 维 + 1 维 特征 ----------
    log.info("=" * 60)
    log.info("Step 5: 构造特征列")
    log.info("=" * 60)

    fy = fy.rename(columns={"Stkcd": "firm_id"})

    # ctrl_separation
    if seperation_col in fy.columns:
        fy["ctrl_separation"] = fy[seperation_col]
    else:
        fy["ctrl_separation"] = np.nan

    # ctrl_ownership / ctrl_voting
    fy["ctrl_ownership"] = fy.get("S0704b", np.nan)
    fy["ctrl_voting"] = fy.get("S0704c", np.nan)

    # ctrl_separation_p90_flag
    fy["ctrl_separation_p90_flag"] = (
        (fy["ctrl_separation"] > P90_SEPERATION_THRESHOLD).fillna(False).astype("int8")
    )

    # ctrl_no_controller_flag: S0701a == 0 (无判定标准) 或 S0701b (实控人姓名) 为空
    no_ctrl = pd.Series(False, index=fy.index)
    if "S0701a" in fy.columns:
        s0701a = pd.to_numeric(fy["S0701a"], errors="coerce")
        no_ctrl = no_ctrl | (s0701a == 0)
    if "S0701b" in fy.columns:
        no_ctrl = no_ctrl | fy["S0701b"].isna() | (fy["S0701b"].astype(str).str.strip() == "")
    fy["ctrl_no_controller_flag"] = no_ctrl.astype("int8")

    # ctrl_state_owned_flag: S0702b 文本含国资关键字
    if "S0702b" in fy.columns:
        s0702b_str = fy["S0702b"].astype(str)
        state_pat = "|".join(STATE_OWNED_KEYWORDS)
        fy["ctrl_state_owned_flag"] = s0702b_str.str.contains(
            state_pat, regex=True, na=False
        ).astype("int8")
    else:
        log.warning("HLD_Contrshr 缺 S0702b 字段, ctrl_state_owned_flag 全置 0")
        fy["ctrl_state_owned_flag"] = 0

    feature_cols = [
        "ctrl_separation",
        "ctrl_ownership",
        "ctrl_voting",
        "ctrl_separation_p90_flag",
        "ctrl_no_controller_flag",
        "ctrl_state_owned_flag",
    ]

    out_df = fy[["firm_id", "year"] + feature_cols].copy()
    log.info(f"聚合后 firm-year 行: {len(out_df)}")

    # ---------- step 6: 与 v0.7 firm-year 对齐, 添加 missingness flag ----------
    log.info("=" * 60)
    log.info("Step 6: 与 v0.7 全集对齐, 添加 missingness flag")
    log.info("=" * 60)

    if base_firm_year_parquet is not None and base_firm_year_parquet.exists():
        base = pd.read_parquet(base_firm_year_parquet, columns=["firm_id", "year"])
        base["firm_id"] = base["firm_id"].astype(str).str.zfill(6)
        base["year"] = base["year"].astype(int)
        base = base.drop_duplicates()
        log.info(f"v0.7 全集: {len(base)} firm-year")

        merged = base.merge(out_df, on=["firm_id", "year"], how="left")

        # missingness flag (v0.7 全集中 HLD_Contrshr 没记录)
        merged["ctrl_data_missing_flag"] = merged["ctrl_separation"].isna().astype("int8")

        # 连续值 fillna(0)
        for c in ["ctrl_separation", "ctrl_ownership", "ctrl_voting"]:
            merged[c] = merged[c].fillna(0).astype("float32")

        # 二值 flag 缺失即未触发 → fillna 0
        for c in ["ctrl_separation_p90_flag", "ctrl_no_controller_flag", "ctrl_state_owned_flag"]:
            merged[c] = merged[c].fillna(0).astype("int8")

        n_miss = int(merged["ctrl_data_missing_flag"].sum())
        log.info(f"v0.7 中 HLD_Contrshr 缺失: {n_miss} firm-year ({n_miss/len(merged)*100:.1f}%)")
        out_df = merged
    else:
        log.warning(f"v0.7 firm-year 全集 {base_firm_year_parquet} 不存在, 跳过对齐. 输出仅 HLD 自身覆盖范围")
        out_df["ctrl_data_missing_flag"] = 0
        for c in ["ctrl_separation", "ctrl_ownership", "ctrl_voting"]:
            out_df[c] = out_df[c].astype("float32")

    # ---------- step 7: 持久化 ----------
    log.info("=" * 60)
    log.info("Step 7: 输出")
    log.info("=" * 60)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    final_cols = ["firm_id", "year"] + feature_cols + ["ctrl_data_missing_flag"]
    out_df[final_cols].to_parquet(out_parquet, index=False, compression="snappy")
    log.info(f"特征写入: {out_parquet}  ({len(out_df)} 行 × {len(final_cols)} 列)")

    # ---------- step 8: 统计 ----------
    stats = {
        "feature_set": "controller (HLD_Contrshr)",
        "input": str(hld_xlsx),
        "output_parquet": str(out_parquet),
        "n_rows_output": int(len(out_df)),
        "n_features": 7,
        "feature_cols": final_cols[2:],
        "feature_distributions": {
            c: {
                "mean": float(out_df[c].mean()),
                "std": float(out_df[c].std()),
                "p50": float(out_df[c].median()),
                "p90": float(out_df[c].quantile(0.9)),
                "max": float(out_df[c].max()),
            }
            for c in feature_cols if c in out_df.columns
        },
        "n_data_missing": int(out_df["ctrl_data_missing_flag"].sum()) if "ctrl_data_missing_flag" in out_df.columns else 0,
        "p90_separation_threshold": P90_SEPERATION_THRESHOLD,
        "state_owned_keywords": STATE_OWNED_KEYWORDS,
        "notes": [
            "ctrl_separation 是论文金牌特征 (La Porta 1999 / Claessens 2002 学术理论支撑)",
            "缺失值已 fillna(0), 同时 ctrl_data_missing_flag 单列保留 missingness 信号",
            "ctrl_state_owned_flag 通过 S0702b 文本含国资关键字判定 (启发式)",
        ],
    }
    write_stats_json(stats, out_stats, log=log)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="HLD_Contrshr → 6 维 + 1 维实控人特征")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--hld-xlsx", type=Path, default=None)
    p.add_argument("--base-firm-year", type=Path, default=None,
                   help="v0.7 features parquet (用于 firm-year 全集对齐)")
    p.add_argument("--out-parquet", type=Path, default=None)
    p.add_argument("--out-stats", type=Path, default=None)
    args = p.parse_args()

    paths = get_paths(args.project_root)
    log = get_logger("integrate_controller_features", paths["stats"])

    hld_xlsx = args.hld_xlsx or (paths["raw_csmar"] / "HLD_Contrshr.xlsx")
    base = args.base_firm_year or (paths["processed_features"] / "features_v0_7.parquet")
    out_parquet = args.out_parquet or (paths["processed_features"] / "controller_features_v1_0.parquet")
    out_stats = args.out_stats or (paths["stats"] / "integrate_controller_stats.json")

    log.info(f"HLD xlsx: {hld_xlsx}")
    log.info(f"v0.7 base: {base}")
    log.info(f"Out: {out_parquet}")

    build_controller_features(hld_xlsx, base, out_parquet, out_stats, log)
    log.info("✓ integrate_controller_features 完成")


if __name__ == "__main__":
    main()
