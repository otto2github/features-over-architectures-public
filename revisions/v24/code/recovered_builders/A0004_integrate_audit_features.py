#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_audit_features.py
===========================
FIN_Audit → 5 维审计特征.

输出 5 维 (firm-year 聚合):
  audit_is_modified           : 二值 (非标准无保留意见 = 1)  ★强信号
  audit_type_code             : categorical 编码 1-9 (Audittyp 中文 → 数字)
  audit_fee_log               : log(Tcost + 1) 审计费对数
  audit_fee_to_assets         : 审计费 / 公司资产 (异常高 = 红旗)
  audit_firm_change_flag      : 二值 (年度间会计师事务所变更, 由 Dadtunit 跨年比较)

CSMAR Audittyp 中文 → 数字编码 (论文需建立映射, 不是数字直接给):
  1 = 标准无保留意见 (Unqualified)
  2 = 带强调事项段的无保留意见 (Unqualified with explanatory paragraph)
  3 = 带解释性说明无保留意见 (Unqualified with emphasis of matter)
  4 = 保留意见 (Qualified)
  5 = 否定意见 (Adverse)
  6 = 拒绝表示意见 (Disclaimer of opinion)
  7 = 无法表示意见 (Disclaimer)  -- 与 6 同义, CSMAR 不同年代措辞差异
  8 = 其他非标准意见
  9 = 未知 / 缺失 (置 1 表示未审计完成)

is_modified = 1 当 type_code != 1 (任何非标准无保留意见均算非标审计)

数据特性 (依据 inventory):
  68,607 行 → 清洁 42,523 行 / 4,107 公司
  非标审计意见占比 ≈ 4.8% (≈ 欺诈率量级, 极强信号)
  Tcost: P50=89.5万, P90=260万

聚合策略:
  - Accper 提取 year (会计年度)
  - 同 firm-year 多条 (年报+审计报告分开): 取 Audittyp 最严重者 (type_code 最大)
  - audit_firm_change_flag: 用 Dadtunit (会计师事务所), 与上年比较
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

# ============================================================ Audittyp 编码映射
AUDIT_TYPE_MAPPING = {
    # type_code: [中文关键词列表 (按出现频率排序)]
    1: ["标准无保留", "无保留意见", "标准审计"],  # 默认 1
    2: ["带强调事项", "强调事项"],
    3: ["带解释性说明", "解释性说明"],
    4: ["保留意见"],
    5: ["否定意见"],
    6: ["拒绝表示", "无法表示"],
    8: ["其他"],
}


def encode_audittyp(s: str) -> int:
    """
    中文 Audittyp 字符串 → 1-9 数字编码 (启发式匹配关键词)

    [BUG FIX 2026-05-08] 原版 for code in [2,3,4,5,6,8] 循环导致
        "标准无保留意见" 中含子串 "保留意见" → 被误判为 code=4 (保留),
        造成 audit_is_modified=99.5% (实际应 ~5-8%).

    判定顺序 (子串包含关系决定的严格优先级):
      "带强调事项段的无保留意见" 含 "无保留" 含 "保留" — 必须先抓 "强调事项"  → 2
      "带解释性说明的无保留意见" 含 "无保留" 含 "保留" — 必须先抓 "解释性说明" → 3
      "否定意见" / "无法表示意见" / "拒绝表示意见" — 互斥, 任意位置抓
      "标准无保留意见" / "无保留意见" — 此时已排除带强调/带解释 → 1
      "保留意见" — 此时已排除 "无保留" → 4 (真正的 Qualified)
    """
    if pd.isna(s) or not isinstance(s, str) or s.strip() == "":
        return 9
    text = str(s).strip()

    # 1. 带强调事项段 (Unqualified with explanatory paragraph) — 优先抓走
    #    注意: CSMAR 实测有"事项段"(无"强调"前缀), 但只出现在"保留意见加事项段",
    #    核心仍是保留, 不在此处抓 — 留给后面的 "保留意见" 兜底.
    if "强调事项" in text or "带强调" in text:
        return 2

    # 2. 带解释性说明 / 说明段 — 修饰版无保留 (CSMAR 1503 准则下 EOM)
    #    "无保留意见加说明段" → 3, 但要排除 "保留意见加说明段" (核心是保留 → 4)
    if "解释性说明" in text or "带解释" in text or "说明段" in text:
        if "保留意见" in text and "无保留" not in text:
            return 4   # 保留意见加说明段
        return 3

    # 3. 否定意见 (Adverse) — 严重非标
    if "否定意见" in text:
        return 5

    # 4. 拒绝 / 无法 (Disclaimer) — CSMAR 实测有 "拒绝发表" / "无法表示" 两种说法
    if "拒绝表示" in text or "拒绝发表" in text or "无法表示" in text or "无法发表" in text:
        return 6

    # 5. 标准无保留 / 无保留 (Unqualified) — 此时已排除带强调/带解释
    if "无保留" in text or ("标准" in text and "审计" in text):
        return 1

    # 6. 保留意见 (Qualified) — 此时已排除"无保留", 是真正的 Qualified
    if "保留意见" in text or "保留" in text:
        return 4

    # 7. 兜底
    return 8


# ============================================================
def build_audit_features(
    audit_xlsx: Path,
    base_firm_year_parquet: Path,
    out_parquet: Path,
    out_stats: Path,
    base_features_for_assets: Path | None,
    log,
):
    # ---------- step 1: 读 FIN_Audit ----------
    log.info("=" * 60)
    log.info("Step 1: 读 FIN_Audit")
    log.info("=" * 60)

    df = read_csmar_xlsx(audit_xlsx, code_col="Stkcd", log=log)
    log.info(f"清洁后行数: {len(df)}")
    log_mem(log, "after_read_audit")

    # ---------- step 2: year ----------
    df["year"] = extract_year(df["Accper"]).astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.rename(columns={"Stkcd": "firm_id"})

    # ---------- step 3: Audittyp 编码 ----------
    log.info("=" * 60)
    log.info("Step 3: Audittyp 中文 → 数字编码")
    log.info("=" * 60)

    df["audit_type_code"] = df["Audittyp"].astype(str).apply(encode_audittyp).astype("int8")

    # 编码分布
    code_dist = df["audit_type_code"].value_counts().to_dict()
    log.info(f"audit_type_code 分布: {dict(sorted({int(k):int(v) for k,v in code_dist.items()}.items()))}")

    # 兜底: 看一下被 encode 为 8 的中文样本, 确认映射没漏
    other_samples = df.loc[df["audit_type_code"] == 8, "Audittyp"].drop_duplicates().head(20).tolist()
    if other_samples:
        log.info(f"被编码为 8 的 Audittyp 样例 (top 20): {other_samples}")

    df["audit_is_modified"] = (df["audit_type_code"] != 1).astype("int8")

    # ---------- step 4: Tcost log + fee_to_assets ----------
    log.info("=" * 60)
    log.info("Step 4: 审计费 log + 审计费率")
    log.info("=" * 60)

    if "Tcost" in df.columns:
        df["Tcost_num"] = pd.to_numeric(df["Tcost"], errors="coerce")
        df["audit_fee_log"] = np.log1p(df["Tcost_num"].fillna(0)).astype("float32")
    else:
        log.warning("FIN_Audit 缺 Tcost 字段")
        df["Tcost_num"] = 0
        df["audit_fee_log"] = 0.0

    # ---------- step 5: 同 firm-year 取最严重 ----------
    log.info("=" * 60)
    log.info("Step 5: firm-year 聚合 (取 audit_type_code 最大者)")
    log.info("=" * 60)

    df_sorted = df.sort_values(["firm_id", "year", "audit_type_code"], ascending=[True, True, False])
    fy = df_sorted.drop_duplicates(["firm_id", "year"], keep="first").copy()
    log.info(f"firm-year 聚合后: {len(fy)} 行")

    # ---------- step 6: audit_firm_change_flag ----------
    log.info("=" * 60)
    log.info("Step 6: 会计师事务所变更标志")
    log.info("=" * 60)

    if "Dadtunit" in fy.columns:
        fy_sorted = fy.sort_values(["firm_id", "year"]).copy()
        fy_sorted["_prev_dadtunit"] = fy_sorted.groupby("firm_id")["Dadtunit"].shift(1)
        fy_sorted["audit_firm_change_flag"] = (
            fy_sorted["_prev_dadtunit"].notna()
            & (fy_sorted["Dadtunit"].astype(str) != fy_sorted["_prev_dadtunit"].astype(str))
        ).astype("int8")
        # 对应回 fy
        fy = fy.merge(
            fy_sorted[["firm_id", "year", "audit_firm_change_flag"]],
            on=["firm_id", "year"],
            how="left",
        )
        fy["audit_firm_change_flag"] = fy["audit_firm_change_flag"].fillna(0).astype("int8")
        n_change = int(fy["audit_firm_change_flag"].sum())
        log.info(f"会计师事务所变更事件: {n_change} firm-year ({n_change/len(fy)*100:.2f}%)")
    else:
        log.warning("FIN_Audit 缺 Dadtunit 字段, audit_firm_change_flag 全置 0")
        fy["audit_firm_change_flag"] = 0

    # ---------- step 7: audit_fee_to_assets (需 join 资产数据) ----------
    log.info("=" * 60)
    log.info("Step 7: audit_fee_to_assets (需 join 总资产)")
    log.info("=" * 60)

    fy["audit_fee_to_assets"] = np.nan  # 默认占位

    if base_features_for_assets is not None and base_features_for_assets.exists():
        # v0.7 features_v0_7.parquet 应该有 total_assets 或类似字段
        # 尝试常见列名
        try:
            base_cols = pd.read_parquet(base_features_for_assets).columns.tolist()
            asset_col = None
            for cand in ["total_assets", "Tot_Assets", "TotalAssets", "assets", "ta_log"]:
                if cand in base_cols:
                    asset_col = cand
                    break
            if asset_col is not None:
                base = pd.read_parquet(base_features_for_assets, columns=["firm_id", "year", asset_col])
                base["firm_id"] = base["firm_id"].astype(str).str.zfill(6)
                base["year"] = base["year"].astype(int)
                base = base.rename(columns={asset_col: "_assets"}).drop_duplicates(["firm_id", "year"])

                fy = fy.merge(base, on=["firm_id", "year"], how="left")
                # 若 base 中是对数, 取 exp 还原 (启发); 一般 ta_log 才需还原
                if asset_col == "ta_log":
                    fy["_assets"] = np.expm1(fy["_assets"])
                # fee / assets, 安全分母
                with np.errstate(divide="ignore", invalid="ignore"):
                    fy["audit_fee_to_assets"] = np.where(
                        fy["_assets"] > 0,
                        fy["Tcost_num"].fillna(0) / fy["_assets"],
                        np.nan,
                    )
                log.info(f"用 v0.7 列 '{asset_col}' 计算 audit_fee_to_assets")
            else:
                log.warning(f"v0.7 features 中未找到资产列 (查询: total_assets/Tot_Assets/...)")
        except Exception as e:
            log.warning(f"读取 v0.7 features 失败: {e}")
    else:
        log.warning("未指定或不存在 base-features-for-assets, audit_fee_to_assets 全 NaN (后续 fillna 0)")

    fy["audit_fee_to_assets"] = pd.to_numeric(fy["audit_fee_to_assets"], errors="coerce").astype("float32")

    # ---------- step 8: 与 v0.7 全集对齐 ----------
    log.info("=" * 60)
    log.info("Step 8: 与 v0.7 全集对齐 (fillna)")
    log.info("=" * 60)

    feature_cols = [
        "audit_is_modified",
        "audit_type_code",
        "audit_fee_log",
        "audit_fee_to_assets",
        "audit_firm_change_flag",
    ]
    out_df = fy[["firm_id", "year"] + feature_cols].copy()

    if base_firm_year_parquet is not None and base_firm_year_parquet.exists():
        base = pd.read_parquet(base_firm_year_parquet, columns=["firm_id", "year"])
        base["firm_id"] = base["firm_id"].astype(str).str.zfill(6)
        base["year"] = base["year"].astype(int)
        base = base.drop_duplicates()

        merged = base.merge(out_df, on=["firm_id", "year"], how="left")
        # 缺失填充 (无审计记录视为 type_code=9, is_modified=0, 费用 0)
        merged["audit_type_code"] = merged["audit_type_code"].fillna(9).astype("int8")
        merged["audit_is_modified"] = merged["audit_is_modified"].fillna(0).astype("int8")
        merged["audit_fee_log"] = merged["audit_fee_log"].fillna(0).astype("float32")
        merged["audit_fee_to_assets"] = merged["audit_fee_to_assets"].fillna(0).astype("float32")
        merged["audit_firm_change_flag"] = merged["audit_firm_change_flag"].fillna(0).astype("int8")

        n_covered = int((merged["audit_type_code"] != 9).sum())
        log.info(f"v0.7 中有审计记录: {n_covered} firm-year ({n_covered/len(merged)*100:.1f}%)")
        out_df = merged

    # ---------- step 9: 持久化 ----------
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False, compression="snappy")
    log.info(f"特征写入: {out_parquet}  ({len(out_df)} 行)")

    stats = {
        "feature_set": "audit (FIN_Audit)",
        "input": str(audit_xlsx),
        "n_rows_output": int(len(out_df)),
        "feature_cols": feature_cols,
        "audit_type_code_distribution": {int(k): int(v) for k, v in out_df["audit_type_code"].value_counts().items()},
        "audit_is_modified_rate": float(out_df["audit_is_modified"].mean()),
        "audit_firm_change_rate": float(out_df["audit_firm_change_flag"].mean()),
        "audit_fee_log_stats": {
            "mean": float(out_df["audit_fee_log"].mean()),
            "p50": float(out_df["audit_fee_log"].median()),
            "p90": float(out_df["audit_fee_log"].quantile(0.9)),
        },
        "audit_type_mapping": {str(k): v for k, v in AUDIT_TYPE_MAPPING.items()},
        "notes": [
            "audit_is_modified == 1 是非标审计意见, 与欺诈率量级 (4.8%) 接近, 强信号",
            "audit_type_code=9 表示未审计/缺失, 占比 = 100% - 覆盖率",
            "audit_fee_to_assets 异常高 (P95+) 是审计师 risk premium, 间接欺诈信号",
        ],
    }
    write_stats_json(stats, out_stats, log=log)


# ============================================================
def main():
    p = argparse.ArgumentParser(description="FIN_Audit → 5 维审计特征")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--audit-xlsx", type=Path, default=None)
    p.add_argument("--base-firm-year", type=Path, default=None,
                   help="v0.7 features parquet (用于 firm-year 全集对齐)")
    p.add_argument("--base-features-for-assets", type=Path, default=None,
                   help="v0.7 features parquet, 含 total_assets 列 (用于计算 fee_to_assets)")
    p.add_argument("--out-parquet", type=Path, default=None)
    p.add_argument("--out-stats", type=Path, default=None)
    args = p.parse_args()

    paths = get_paths(args.project_root)
    log = get_logger("integrate_audit_features", paths["stats"])

    audit_xlsx = args.audit_xlsx or (paths["raw_csmar"] / "FIN_Audit.xlsx")
    base = args.base_firm_year or (paths["processed_features"] / "features_v0_7.parquet")
    base_assets = args.base_features_for_assets or (paths["processed_features"] / "features_v0_7.parquet")
    out_parquet = args.out_parquet or (paths["processed_features"] / "audit_features_v1_0.parquet")
    out_stats = args.out_stats or (paths["stats"] / "integrate_audit_stats.json")

    log.info(f"Audit xlsx: {audit_xlsx}")
    log.info(f"Base firm-year: {base}")
    log.info(f"Base for assets: {base_assets}")
    log.info(f"Out: {out_parquet}")

    build_audit_features(audit_xlsx, base, out_parquet, out_stats, base_assets, log)
    log.info("✓ integrate_audit_features 完成")


if __name__ == "__main__":
    main()
