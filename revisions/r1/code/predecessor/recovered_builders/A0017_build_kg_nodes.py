"""
build_kg_nodes.py
=================

构建知识图谱节点表 (Chapter 7 GNN).

节点类型:
  C:<ts_code>           上市公司 (来自 stk_basic.parquet 或 ts_code 集合)
  P:<sha1[:12]>         自然人 (来自 stk_managers + top10_holders 中判定为 person 的 holder)
  I:<sha1[:12]>         机构 (来自 top10_holders + top10_floatholders 中判定为 institution 的 holder)

输出:
  data/processed/kg/nodes.parquet              全部节点 (id, type, canonical_name, raw_name_first, source, n_records)
  data/processed/kg/companies.parquet          公司节点子集 (id, ts_code)
  data/processed/kg/persons.parquet            自然人节点子集
  data/processed/kg/institutions.parquet       机构节点子集
  data/interim/kg_stats/build_kg_nodes.json    统计 (各类型节点数 / 来源分布 / 边界情形)

用法:
  uv run --project ~/code python scripts/data_collect/kg/build_kg_nodes.py

设计:
  - source 字段记录来自哪个 raw 数据源 (top10_holders / top10_floatholders / stk_managers / 多源合并)
  - n_records 字段记录该节点在 raw 中出现总次数, 用于剔除"一次性"低质量节点
  - 边界情形 (canonical 长度异常 / 有疑义) 写入 review_queue.csv 供人工核对
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _kg_common import (  # noqa: E402
    KG_DIR,
    STATS_DIR,
    canonicalize_holder,
    classify_entity,
    get_paths,
    make_company_id,
    make_institution_id,
    make_person_id,
    setup_logger,
    write_stats_json,
)

TASK = "build_kg_nodes"


# ============================================================================
# 加载原始 parquet 集合
# ============================================================================

def load_partitions(d: Path, logger) -> pd.DataFrame:
    """读 d 目录下所有 *.parquet (跳过 _all/_*.parquet 前缀)."""
    if not d.exists():
        logger.error("目录不存在: %s", d)
        return pd.DataFrame()
    parts = sorted(p for p in d.glob("*.parquet")
                   if not p.name.startswith("_") and ".tmp" not in p.name)
    if not parts:
        logger.error("无 parquet 文件: %s", d)
        return pd.DataFrame()
    dfs = []
    for p in parts:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            logger.warning("读取失败 %s: %s", p, e)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    logger.info("[%s] 合并 %d 个分片, 共 %d 行 × %d 列",
                d.name, len(parts), len(df), len(df.columns))
    return df


# ============================================================================
# 公司节点
# ============================================================================

def build_company_nodes(holders_df, float_df, mgr_df, logger) -> pd.DataFrame:
    """
    公司节点 = 三个数据源里出现过的全部 ts_code.
    """
    ts_codes = set()
    for src_name, df in [("top10_holders", holders_df),
                         ("top10_floatholders", float_df),
                         ("stk_managers", mgr_df)]:
        if df is None or df.empty or "ts_code" not in df.columns:
            continue
        ts_codes |= set(df["ts_code"].dropna().unique())
    ts_codes = sorted(ts_codes)
    rows = []
    for tc in ts_codes:
        rows.append({
            "id": make_company_id(tc),
            "type": "company",
            "ts_code": tc,
            "canonical_name": tc,
            "raw_name_first": tc,
            "source": "ts_code_set",
            "n_records": 1,
        })
    df = pd.DataFrame(rows)
    logger.info("公司节点: %d", len(df))
    return df


# ============================================================================
# 持有人节点 (top10_holders + top10_floatholders)
# ============================================================================

def build_holder_nodes(holders_df, float_df, logger) -> pd.DataFrame:
    """
    从两张 holders 表抽 holder_name → 规范化 + 分类.
    返回 (id, type=person/institution, canonical_name, raw_name_first, source, n_records).
    """
    parts = []
    for src_name, df in [("top10_holders", holders_df),
                         ("top10_floatholders", float_df)]:
        if df is None or df.empty:
            continue
        if "holder_name" not in df.columns:
            logger.warning("[%s] 无 holder_name 列, 跳过", src_name)
            continue
        sub = df[["holder_name"]].copy()
        sub["src"] = src_name
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    cat = pd.concat(parts, ignore_index=True)
    cat["holder_name"] = cat["holder_name"].astype(str)

    # 规范化
    logger.info("规范化 %d 条 holder_name ...", len(cat))
    cat["canonical"] = cat["holder_name"].map(canonicalize_holder)
    cat = cat[cat["canonical"].astype(bool)].reset_index(drop=True)

    # 分类
    cat["entity_type"] = cat.apply(
        lambda r: classify_entity(r["canonical"], raw_name=r["holder_name"]),
        axis=1,
    )

    # 聚合: 每个 canonical 一个节点, 记录首次出现的 raw_name + 出现总次数
    grp = (cat.groupby(["canonical", "entity_type"])
              .agg(raw_name_first=("holder_name", "first"),
                   n_records=("holder_name", "count"),
                   sources=("src", lambda s: ",".join(sorted(set(s)))))
              .reset_index())

    # ID
    def _make_id(row):
        if row["entity_type"] == "person":
            return make_person_id(row["canonical"])
        return make_institution_id(row["canonical"])

    grp["id"] = grp.apply(_make_id, axis=1)
    grp = grp.rename(columns={
        "canonical": "canonical_name",
        "entity_type": "type",
        "sources": "source",
    })
    grp = grp[["id", "type", "canonical_name", "raw_name_first",
               "source", "n_records"]]

    persons = (grp["type"] == "person").sum()
    insts = (grp["type"] == "institution").sum()
    logger.info("持有人节点: 自然人 %d, 机构 %d (合计 %d)",
                persons, insts, len(grp))
    return grp


# ============================================================================
# 高管节点 (stk_managers)
# ============================================================================

def build_manager_nodes(mgr_df, logger) -> pd.DataFrame:
    """
    高管几乎全部为自然人 (name 列). 用 classify_entity 兜底, 但绝大多数应为 person.
    """
    if mgr_df is None or mgr_df.empty:
        return pd.DataFrame()
    if "name" not in mgr_df.columns:
        logger.warning("stk_managers 无 name 列, 跳过")
        return pd.DataFrame()

    sub = mgr_df[["name"]].copy()
    sub["name"] = sub["name"].astype(str)
    sub["canonical"] = sub["name"].map(canonicalize_holder)
    sub = sub[sub["canonical"].astype(bool)].reset_index(drop=True)
    sub["entity_type"] = sub.apply(
        lambda r: classify_entity(r["canonical"], raw_name=r["name"]),
        axis=1,
    )

    grp = (sub.groupby(["canonical", "entity_type"])
              .agg(raw_name_first=("name", "first"),
                   n_records=("name", "count"))
              .reset_index())

    def _make_id(row):
        if row["entity_type"] == "person":
            return make_person_id(row["canonical"])
        return make_institution_id(row["canonical"])

    grp["id"] = grp.apply(_make_id, axis=1)
    grp = grp.rename(columns={
        "canonical": "canonical_name",
        "entity_type": "type",
    })
    grp["source"] = "stk_managers"
    grp = grp[["id", "type", "canonical_name", "raw_name_first",
               "source", "n_records"]]

    persons = (grp["type"] == "person").sum()
    insts = (grp["type"] == "institution").sum()
    logger.info("高管节点: 自然人 %d, 机构 %d (合计 %d)",
                persons, insts, len(grp))
    return grp


# ============================================================================
# 合并 + 去重 (跨数据源同 id 应合并)
# ============================================================================

def merge_node_tables(*tables, logger) -> pd.DataFrame:
    valid = [t for t in tables if t is not None and not t.empty]
    if not valid:
        return pd.DataFrame()
    cat = pd.concat(valid, ignore_index=True)

    # 同 id 合并 (跨数据源)
    grp = (cat.groupby("id")
              .agg(type=("type", "first"),
                   canonical_name=("canonical_name", "first"),
                   raw_name_first=("raw_name_first", "first"),
                   source=("source", lambda s: ",".join(sorted(set(",".join(s).split(","))))),
                   n_records=("n_records", "sum"))
              .reset_index())
    logger.info("合并后节点总数: %d", len(grp))
    logger.info("  按 type 分布: %s",
                dict(grp["type"].value_counts()))
    return grp


# ============================================================================
# 边界情形审核队列
# ============================================================================

def export_review_queue(nodes_df: pd.DataFrame, logger) -> int:
    """
    导出可疑节点供人工复核:
      - canonical_name 长度 ≤ 1 (异常短)
      - canonical_name 长度 > 30 (异常长)
      - n_records == 1 且类型为 institution (一次性出现, 可能是数据错误)
      - 自然人但 raw_name_first 含数字 / ASCII (可能误判)
    """
    issues = []
    for _, r in nodes_df.iterrows():
        if r["type"] == "company":
            continue
        cn = r["canonical_name"]
        rn = r["raw_name_first"]
        n = r["n_records"]
        flags = []
        if len(cn) <= 1:
            flags.append("too_short")
        if len(cn) > 30:
            flags.append("too_long")
        if r["type"] == "institution" and n == 1:
            flags.append("singleton_inst")
        if r["type"] == "person":
            import re as _re
            if _re.search(r"[A-Za-z0-9]", rn or ""):
                flags.append("person_has_ascii")
            if len(cn) > 6:
                flags.append("person_too_long")
        if flags:
            issues.append({
                "id": r["id"],
                "type": r["type"],
                "canonical_name": cn,
                "raw_name_first": rn,
                "n_records": n,
                "flags": ",".join(flags),
            })
    if not issues:
        logger.info("审核队列: 无可疑节点")
        return 0
    df = pd.DataFrame(issues)
    out = STATS_DIR / "build_kg_nodes_review_queue.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("审核队列: %d 条, 写入 %s", len(df), out)
    return len(df)


# ============================================================================
# 主流程
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-export", action="store_true",
                   help="跳过 parquet 输出 (仅做统计验证)")
    args = p.parse_args()

    logger = setup_logger(TASK)
    paths = get_paths()

    # ---- 1. 读三个原始数据源 ----
    holders = load_partitions(paths["TOP10_HOLDERS_DIR"], logger)
    floaters = load_partitions(paths["TOP10_FLOAT_DIR"], logger)
    mgrs = load_partitions(paths["STK_MANAGERS_DIR"], logger)

    if holders.empty and floaters.empty and mgrs.empty:
        logger.error("三个数据源都为空, 退出")
        return 1

    # ---- 2. 构造各类节点 ----
    company_nodes = build_company_nodes(holders, floaters, mgrs, logger)
    holder_nodes = build_holder_nodes(holders, floaters, logger)
    mgr_nodes = build_manager_nodes(mgrs, logger)

    # ---- 3. 合并 ----
    all_nodes = merge_node_tables(company_nodes, holder_nodes, mgr_nodes,
                                  logger=logger)

    # ---- 4. 审核队列 ----
    n_review = export_review_queue(all_nodes, logger)

    # ---- 5. 统计 ----
    stats = {
        "raw": {
            "top10_holders_rows": int(len(holders)),
            "top10_float_rows": int(len(floaters)),
            "stk_managers_rows": int(len(mgrs)),
        },
        "nodes": {
            "total": int(len(all_nodes)),
            "by_type": {k: int(v) for k, v in
                        all_nodes["type"].value_counts().to_dict().items()},
            "n_records_distribution": {
                "p50": int(all_nodes["n_records"].quantile(0.5)),
                "p90": int(all_nodes["n_records"].quantile(0.9)),
                "p99": int(all_nodes["n_records"].quantile(0.99)),
                "max": int(all_nodes["n_records"].max()),
            },
            "review_queue_size": n_review,
        },
        "company_count_check": {
            "expected_5341_or_so": True,
            "actual": int((all_nodes["type"] == "company").sum()),
        },
    }
    write_stats_json(TASK, stats)
    logger.info("统计已写入 %s/%s.json", STATS_DIR, TASK)

    # ---- 6. 输出 parquet ----
    if not args.no_export:
        all_nodes.to_parquet(KG_DIR / "nodes.parquet", index=False, compression="snappy")
        company_nodes.to_parquet(KG_DIR / "companies.parquet", index=False, compression="snappy")
        persons = all_nodes[all_nodes["type"] == "person"]
        insts = all_nodes[all_nodes["type"] == "institution"]
        persons.to_parquet(KG_DIR / "persons.parquet", index=False, compression="snappy")
        insts.to_parquet(KG_DIR / "institutions.parquet", index=False, compression="snappy")
        logger.info("nodes.parquet 等已写入 %s", KG_DIR)

    logger.info("=" * 60)
    logger.info("build_kg_nodes 完成")
    logger.info("  公司:   %d", int((all_nodes["type"] == "company").sum()))
    logger.info("  自然人: %d", int((all_nodes["type"] == "person").sum()))
    logger.info("  机构:   %d", int((all_nodes["type"] == "institution").sum()))
    logger.info("  审核队列: %d", n_review)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
