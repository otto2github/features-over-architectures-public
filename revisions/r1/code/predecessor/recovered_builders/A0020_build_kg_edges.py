"""
build_kg_edges.py
=================

构建 5 种边类型 (Chapter 7 GNN).

边类型 (按论文重要性排序):
  E1. 公司 → 持有人 (HOLDS_BY)            from top10_holders     attrs: hold_ratio, year
  E2. 公司 → 流通持有人 (FLOAT_HELD_BY)    from top10_floatholders attrs: hold_ratio, year
  E3. 公司 → 高管 (HAS_MANAGER)            from stk_managers       attrs: lev, title, year_active
  E4. 公司 ↔ 公司 派生: 共同股东 (CO_HELD)  E1+E2 派生              attrs: shared_holder_id, year, weight
  E5. 公司 ↔ 公司 派生: 共同高管 (CO_MGR)   E3 派生                 attrs: shared_person_id, year

频度统一为年频:
  E1/E2: 季频 → 取每年最后季度的 hold_ratio (年末快照).
  E3:   按高管任职区间转换为 (公司, 自然人, year) 多年记录 (1 个高管 N 年 → N 行).
  E4/E5: 派生为年度边 (按 year 切片).

输出:
  data/processed/kg/edges_company_holder.parquet
  data/processed/kg/edges_company_float_holder.parquet
  data/processed/kg/edges_company_manager.parquet
  data/processed/kg/edges_company_company_co_held.parquet
  data/processed/kg/edges_company_company_co_mgr.parquet
  data/interim/kg_stats/build_kg_edges.json

用法:
  uv run --project ~/code python scripts/data_collect/kg/build_kg_edges.py
  --start-year 2010 --end-year 2024
  --co-held-min-shared 1   # 共同股东派生时, 至少共享 N 个 holder 才连边
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _kg_common import (  # noqa: E402
    KG_DIR,
    canonicalize_holder,
    classify_entity,
    get_paths,
    make_company_id,
    make_institution_id,
    make_person_id,
    setup_logger,
    write_stats_json,
)

TASK = "build_kg_edges"


# ============================================================================
# 工具
# ============================================================================

def load_partitions(d: Path, logger) -> pd.DataFrame:
    if not d.exists():
        return pd.DataFrame()
    parts = sorted(p for p in d.glob("*.parquet")
                   if not p.name.startswith("_") and ".tmp" not in p.name)
    if not parts:
        return pd.DataFrame()
    dfs = []
    for p in parts:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            logger.warning("读取失败 %s: %s", p, e)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def add_holder_id(df: pd.DataFrame) -> pd.DataFrame:
    """给 holder df 添加规范化的 canonical / entity_type / holder_id."""
    df = df.copy()
    df["holder_name"] = df["holder_name"].astype(str)
    df["canonical"] = df["holder_name"].map(canonicalize_holder)
    df = df[df["canonical"].astype(bool)].reset_index(drop=True)
    df["entity_type"] = df.apply(
        lambda r: classify_entity(r["canonical"], raw_name=r["holder_name"]),
        axis=1,
    )
    df["holder_id"] = df.apply(
        lambda r: (make_person_id(r["canonical"])
                   if r["entity_type"] == "person"
                   else make_institution_id(r["canonical"])),
        axis=1,
    )
    return df


def add_period_year(df: pd.DataFrame) -> pd.DataFrame:
    """从 end_date / period 提取 year."""
    df = df.copy()
    if "end_date" in df.columns:
        df["year"] = pd.to_datetime(df["end_date"], format="%Y%m%d",
                                     errors="coerce").dt.year
    elif "period" in df.columns:
        df["year"] = pd.to_datetime(df["period"], format="%Y%m%d",
                                     errors="coerce").dt.year
    elif "ann_date" in df.columns:
        df["year"] = pd.to_datetime(df["ann_date"], format="%Y%m%d",
                                     errors="coerce").dt.year
    else:
        df["year"] = np.nan
    return df


# ============================================================================
# E1 / E2: 公司 → 持有人
# ============================================================================

def build_company_holder_edges(holders_df, year_col_for_snap, logger,
                                edge_type: str) -> pd.DataFrame:
    """
    季频 → 年末快照 (取每年 max(end_date) 的记录).
    """
    if holders_df is None or holders_df.empty:
        return pd.DataFrame()
    df = add_period_year(holders_df)
    df = df.dropna(subset=["year"]).reset_index(drop=True)
    df["year"] = df["year"].astype(int)

    # 年末快照: 同 (ts_code, year, holder_name) 取最后一个 end_date
    if "end_date" not in df.columns:
        logger.warning("[%s] 无 end_date, 无法做年末快照", edge_type)
        return pd.DataFrame()

    df = df.sort_values(["ts_code", "year", "holder_name", "end_date"])
    snap = df.drop_duplicates(subset=["ts_code", "year", "holder_name"], keep="last")

    # 加 holder_id
    snap = add_holder_id(snap)

    # 抽边
    cols = ["ts_code", "year", "holder_id", "entity_type"]
    keep = ["hold_ratio"] if "hold_ratio" in snap.columns else []
    if "hold_amount" in snap.columns:
        keep.append("hold_amount")
    out = snap[cols + keep].copy()
    out["src"] = out["ts_code"].map(make_company_id)
    out["dst"] = out["holder_id"]
    out["edge_type"] = edge_type
    out = out[["src", "dst", "year", "edge_type", "entity_type"] + keep]
    logger.info("[%s] 抽出边 %d 行 (年末快照)", edge_type, len(out))
    return out


# ============================================================================
# E3: 公司 → 高管 (按任职年份展开)
# ============================================================================

def build_company_manager_edges(mgr_df, start_year, end_year, logger) -> pd.DataFrame:
    """
    stk_managers: ts_code, name, lev, title, begin_date, end_date
    按 [begin_date.year, end_date.year] 展开为多年记录.
    """
    if mgr_df is None or mgr_df.empty:
        return pd.DataFrame()
    df = mgr_df.copy()
    df["name"] = df["name"].astype(str)
    df["canonical"] = df["name"].map(canonicalize_holder)
    df = df[df["canonical"].astype(bool)].reset_index(drop=True)
    df["entity_type"] = df.apply(
        lambda r: classify_entity(r["canonical"], raw_name=r["name"]),
        axis=1,
    )
    df["mgr_id"] = df.apply(
        lambda r: (make_person_id(r["canonical"])
                   if r["entity_type"] == "person"
                   else make_institution_id(r["canonical"])),
        axis=1,
    )

    # 解析日期
    def parse_year(x, default):
        if pd.isna(x) or x is None or x == "":
            return default
        try:
            return int(str(x)[:4])
        except (ValueError, TypeError):
            return default

    df["begin_year"] = df["begin_date"].map(lambda x: parse_year(x, start_year))
    df["end_year"] = df["end_date"].map(lambda x: parse_year(x, end_year))
    # clip
    df["begin_year"] = df["begin_year"].clip(start_year, end_year)
    df["end_year"] = df["end_year"].clip(start_year, end_year)

    # 展开
    rows = []
    for _, r in df.iterrows():
        by, ey = r["begin_year"], r["end_year"]
        if by > ey:
            continue
        for y in range(by, ey + 1):
            rows.append({
                "src": make_company_id(r["ts_code"]),
                "dst": r["mgr_id"],
                "year": y,
                "edge_type": "HAS_MANAGER",
                "entity_type": r["entity_type"],
                "lev": r.get("lev"),
                "title": r.get("title"),
            })
    out = pd.DataFrame(rows)
    logger.info("[HAS_MANAGER] 展开后边 %d 行 (年份维度)", len(out))
    return out


# ============================================================================
# E4 / E5: 公司 ↔ 公司 派生边
# ============================================================================

def derive_company_company_edges(company_holder_df: pd.DataFrame,
                                  edge_type: str,
                                  attr_name: str,
                                  min_shared: int,
                                  logger) -> pd.DataFrame:
    """
    通过 (company, holder, year) 派生 (company_a, company_b, year, shared_holder, weight).
    """
    if company_holder_df is None or company_holder_df.empty:
        return pd.DataFrame()
    df = company_holder_df[["src", "dst", "year"]].copy()
    df.columns = ["company", "holder", "year"]

    # 同 (holder, year) groupby, 笛卡尔积公司 → 公司对
    pairs = []
    grp = df.groupby(["holder", "year"])
    n_groups = len(grp)
    logger.info("[%s] 共 %d 个 (holder, year) 组待派生", edge_type, n_groups)
    cnt = 0
    skipped_huge = 0
    for (holder, year), g in grp:
        cs = g["company"].unique()
        if len(cs) < 2:
            continue
        # 大型 holder (e.g. 香港中央结算) 持有数百上市公司, 笛卡尔积爆炸 → 跳过
        if len(cs) > 50:
            skipped_huge += 1
            continue
        cs_sorted = sorted(cs)
        for i, ca in enumerate(cs_sorted):
            for cb in cs_sorted[i + 1:]:
                pairs.append((ca, cb, year, holder))
        cnt += 1

    logger.info("[%s] 派生 %d 个公司对, 跳过 %d 个大型 holder (>50 公司)",
                edge_type, len(pairs), skipped_huge)

    if not pairs:
        return pd.DataFrame()

    out = pd.DataFrame(pairs, columns=["src", "dst", "year", attr_name])
    # weight = 同 (src, dst, year) 共享 holder 数
    weight = (out.groupby(["src", "dst", "year"])
                 .size().reset_index(name="weight"))
    out_agg = (out.groupby(["src", "dst", "year"])
                  [attr_name].apply(lambda s: ",".join(map(str, sorted(set(s)))))
                  .reset_index())
    out_agg = out_agg.merge(weight, on=["src", "dst", "year"])
    out_agg["edge_type"] = edge_type
    out_agg = out_agg[out_agg["weight"] >= min_shared]
    logger.info("[%s] 聚合后公司对 %d 条 (min_shared=%d)",
                edge_type, len(out_agg), min_shared)
    return out_agg


# ============================================================================
# 主流程
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--co-held-min-shared", type=int, default=1,
                   help="共同股东派生时, 至少共享几个 holder 才连边")
    p.add_argument("--co-held-skip-large", type=int, default=50,
                   help="跳过持有 N 家以上公司的大型 holder (中央结算 / 汇金等)")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--no-derived", action="store_true",
                   help="跳过 E4/E5 派生边 (节省内存, 仅输出 E1/E2/E3)")
    args = p.parse_args()

    logger = setup_logger(TASK)
    paths = get_paths()

    # ---- 1. 加载 ----
    logger.info("加载原始数据 ...")
    holders = load_partitions(paths["TOP10_HOLDERS_DIR"], logger)
    floaters = load_partitions(paths["TOP10_FLOAT_DIR"], logger)
    mgrs = load_partitions(paths["STK_MANAGERS_DIR"], logger)
    logger.info("  top10_holders:      %d 行", len(holders))
    logger.info("  top10_floatholders: %d 行", len(floaters))
    logger.info("  stk_managers:       %d 行", len(mgrs))

    # ---- 2. E1 / E2 / E3 ----
    e1 = build_company_holder_edges(holders, "year", logger, "HOLDS_BY")
    e2 = build_company_holder_edges(floaters, "year", logger, "FLOAT_HELD_BY")
    e3 = build_company_manager_edges(mgrs, args.start_year, args.end_year, logger)

    # ---- 3. E4 / E5 派生 ----
    e4 = pd.DataFrame()
    e5 = pd.DataFrame()
    if not args.no_derived:
        # E4: 共同股东 (基于 E1, 因 E1 含全部 top10_holders)
        e4 = derive_company_company_edges(
            e1, "CO_HELD", "shared_holder",
            args.co_held_min_shared, logger,
        )
        # E5: 共同高管 (基于 E3 中 person 类型)
        e3_person = e3[e3["entity_type"] == "person"] if not e3.empty else pd.DataFrame()
        e5 = derive_company_company_edges(
            e3_person, "CO_MGR", "shared_person",
            1, logger,
        )

    # ---- 4. 输出 ----
    if not args.no_export:
        e1.to_parquet(KG_DIR / "edges_company_holder.parquet", index=False, compression="snappy")
        e2.to_parquet(KG_DIR / "edges_company_float_holder.parquet", index=False, compression="snappy")
        e3.to_parquet(KG_DIR / "edges_company_manager.parquet", index=False, compression="snappy")
        if not args.no_derived:
            e4.to_parquet(KG_DIR / "edges_company_company_co_held.parquet", index=False, compression="snappy")
            e5.to_parquet(KG_DIR / "edges_company_company_co_mgr.parquet", index=False, compression="snappy")
        logger.info("edges_*.parquet 已写入 %s", KG_DIR)

    # ---- 5. 统计 ----
    stats = {
        "params": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "co_held_min_shared": args.co_held_min_shared,
            "co_held_skip_large": args.co_held_skip_large,
        },
        "edge_counts": {
            "E1_HOLDS_BY":      int(len(e1)),
            "E2_FLOAT_HELD_BY": int(len(e2)),
            "E3_HAS_MANAGER":   int(len(e3)),
            "E4_CO_HELD":       int(len(e4)),
            "E5_CO_MGR":        int(len(e5)),
            "total":            int(len(e1) + len(e2) + len(e3) + len(e4) + len(e5)),
        },
        "year_distribution_E1": (
            dict(e1.groupby("year").size().to_dict()) if not e1.empty else {}
        ),
        "year_distribution_E3": (
            dict(e3.groupby("year").size().to_dict()) if not e3.empty else {}
        ),
        "co_held_weight_p50_p90_max": (
            {"p50": int(e4["weight"].quantile(0.5)),
             "p90": int(e4["weight"].quantile(0.9)),
             "max": int(e4["weight"].max())} if not e4.empty else {}
        ),
    }
    write_stats_json(TASK, stats)
    logger.info("统计写入 %s/%s.json", paths["STATS_DIR"], TASK)
    logger.info("=" * 60)
    logger.info("build_kg_edges 完成. 各类型边总数:")
    for k, v in stats["edge_counts"].items():
        logger.info("  %-20s %d", k, v)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
