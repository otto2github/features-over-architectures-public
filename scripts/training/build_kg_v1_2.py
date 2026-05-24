"""
build_kg_v1_2.py
==================

Heterogeneous graph extension: augments v0_8 KG (E1-E5) with E6 (RPT_Repaco) + E7 (RPT_Operation),
build KG v1.2.

Input:
  data/processed/kg/global_edge_index.parquet     (4.47M, E1-E5, 5 edge types)
  data/processed/kg/edges_company_party_trade_v1_0.parquet  (E7, 1.14M, RPT_Operation)
  [optional] data/processed/kg/edges_company_party_repaco_v1_0.parquet (E6, RPT_Repaco)
  data/processed/kg/node_id_index.parquet         (old node-id index)

Output:
  data/processed/kg/global_edge_index_v1_2.parquet (merged, ~5.6M+ edges)
  data/processed/kg/node_id_index_v1_2.parquet    (new node-id index including added related parties)
  data/interim/v1_integration_stats/build_kg_v1_2_stats.json

KG schema:
  - E1 company → major shareholder holds (HOLDS_BY)
  - E2 company → floating-share shareholder (FLOAT_HELD_BY)
  - E3 company → manager (HAS_MANAGER)
  - E4 co-holding relation
  - E5 co-management relation
  - E6 company → related party receivable/payable (REPACO)         [if data available]
  - E7 company → related-party operating transactions (TRADE_*)        [new, 5 sub-types: FUND/GUARANTEE/COMMERCIAL/ASSET/OTHER]

Usage:
  uv run --project ${PROJECT_ROOT:-~/code/foa_project}/.venv python build_kg_v1_2.py \\
      --project-root ${PROJECT_ROOT:-~/code/foa_project}
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


def setup_logger(name, log_dir):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    args = p.parse_args()

    proj = args.project_root or (Path.home() / "code" / "foa_project")
    if not proj.exists():
        print(f"project root does not exist: {proj}", file=sys.stderr)
        return 1

    kg_dir = proj / "data" / "processed" / "kg"
    log_dir = proj / "data" / "interim" / "v1_integration_stats"
    log = setup_logger("build_kg_v1_2", log_dir)

    log.info("=" * 60)
    log.info("KG v1.2 construction (E1-E5 + E7 [+ E6 if exists])")
    log.info("=" * 60)

    # ---------- Input ----------
    edges_v0_8_path = kg_dir / "global_edge_index.parquet"
    edges_e7_path = kg_dir / "edges_company_party_trade_v1_0.parquet"
    edges_e6_path = kg_dir / "edges_company_party_repaco_v1_0.parquet"  # may not exist
    node_idx_path = kg_dir / "node_id_index.parquet"

    if not edges_v0_8_path.exists():
        log.error("v0_8 KG does not exist: %s", edges_v0_8_path)
        return 2
    if not edges_e7_path.exists():
        log.error("E7 edges file does not exist: %s", edges_e7_path)
        return 2

    # ---------- step 1: load base files ----------
    log.info("Step 1: reading v0_8 KG (E1-E5)")
    edges_old = pd.read_parquet(edges_v0_8_path)
    node_idx_old = pd.read_parquet(node_idx_path)
    log.info("  edges_v0_8: %d edges", len(edges_old))
    log.info("  node_idx:   %d nodes", len(node_idx_old))
    log.info("  edge_type distribution: %s", edges_old["edge_type"].value_counts().to_dict())

    # ---------- step 2: load E7 (RPT_Operation) ----------
    log.info("Step 2: load E7 (RPT_Operation related-party transactionedges)")
    e7 = pd.read_parquet(edges_e7_path)
    log.info("  E7 raw: %d edges", len(e7))
    log.info("  E7 columns: %s", e7.columns.tolist())
    log.info("  E7 edge_type: %s", e7["edge_type"].value_counts().to_dict())

    # E7 uses src_node_id (str) / dst_node_id (str)
    # convert src/dst to integer idx, to be compatible with v0_8 KG's src_idx/dst_idx

    # ---------- step 3: node indexextended ----------
    log.info("Step 3: extended node index(new related parties)")
    existing_node_ids = set(node_idx_old["node_id"].astype(str))
    log.info("  existing nodes: %d", len(existing_node_ids))

    e7_src_ids = set(e7["src_node_id"].astype(str))
    e7_dst_ids = set(e7["dst_node_id"].astype(str))
    new_dst_ids = e7_dst_ids - existing_node_ids
    new_src_ids = e7_src_ids - existing_node_ids   # should be 0 (all companies already in v0_8)
    log.info("  E7 dst new nodes: %d", len(new_dst_ids))
    log.info("  E7 src not in v0_8: %d (should be 0)", len(new_src_ids))

    # assign idx to new nodes
    next_idx = node_idx_old["node_idx"].max() + 1
    new_nodes = []
    for nid in sorted(new_dst_ids):
        node_type = "related_party"
        if nid.startswith("PID:"):
            node_type = "related_party"
            ts_code = ""
        elif nid.startswith("NM:"):
            node_type = "related_party_named"
            ts_code = ""
        else:
            node_type = "unknown"
            ts_code = ""
        new_nodes.append({
            "node_idx": next_idx,
            "node_id": nid,
            "node_type": node_type,
            "ts_code": ts_code,
        })
        next_idx += 1
    if new_src_ids:
        log.warning("  E7 contains %d src not in v0_8 company-node table; these edges will be discarded",
                    len(new_src_ids))

    new_nodes_df = pd.DataFrame(new_nodes)
    node_idx_new = pd.concat([node_idx_old, new_nodes_df], ignore_index=True)
    log.info("  extended nodes: %d (added %d)", len(node_idx_new), len(new_nodes_df))

    # ---------- step 4: E7 edges ID → idx ----------
    log.info("Step 4: E7 edges ID → idx mapping")
    nid_to_idx = dict(zip(node_idx_new["node_id"].astype(str),
                           node_idx_new["node_idx"].astype(int)))

    # filter: src must be in company nodes
    e7 = e7[e7["src_node_id"].astype(str).isin(existing_node_ids)].copy()
    log.info("  E7 after filtering: %d edges", len(e7))

    e7["src_idx"] = e7["src_node_id"].astype(str).map(nid_to_idx)
    e7["dst_idx"] = e7["dst_node_id"].astype(str).map(nid_to_idx)

    n_unmapped = e7[["src_idx", "dst_idx"]].isna().any(axis=1).sum()
    if n_unmapped > 0:
        log.warning("  E7 edges: %d rows failed to map", n_unmapped)
        e7 = e7.dropna(subset=["src_idx", "dst_idx"])

    # E7 edge_type renamed to E7_TRADE_{original type}
    e7["edge_type_v12"] = "E7_TRADE_" + e7["edge_type"].astype(str)

    # Output v0_8-compatible format: src_idx, dst_idx, edge_type, year
    e7_compat = pd.DataFrame({
        "src_idx": e7["src_idx"].astype("int64"),
        "dst_idx": e7["dst_idx"].astype("int64"),
        "edge_type": e7["edge_type_v12"].astype(str),
        "year": e7["year"].astype("int64"),
    })
    log.info("  E7 compatibility-format edges: %d", len(e7_compat))
    log.info("  E7 edge_type sub-types: %s", e7_compat["edge_type"].value_counts().to_dict())

    # ---------- step 5: E6 (optional) ----------
    e6_compat = None
    if edges_e6_path.exists():
        log.info("Step 5: load E6 (RPT_Repaco receivable/payable)")
        e6 = pd.read_parquet(edges_e6_path)
        log.info("  E6 raw: %d", len(e6))
        # same handling...(skip if E6 file is not yet generated)
    else:
        log.info("Step 5: E6 (RPT_Repaco) file does not exist; skipping (this run uses only E1-E5 + E7)")

    # ---------- step 6: merge + persist ----------
    log.info("Step 6: merge v0_8 + E7")
    edges_v12 = pd.concat([edges_old, e7_compat], ignore_index=True)
    log.info("  merged edges: %d (E1-E5: %d, E7: %d)",
             len(edges_v12), len(edges_old), len(e7_compat))
    log.info("  merged edge_type distribution:")
    for t, n in edges_v12["edge_type"].value_counts().items():
        log.info("    %-25s %d", t, n)

    out_edges = kg_dir / "global_edge_index_v1_2.parquet"
    out_nodes = kg_dir / "node_id_index_v1_2.parquet"
    edges_v12.to_parquet(out_edges, index=False, compression="snappy")
    node_idx_new.to_parquet(out_nodes, index=False, compression="snappy")
    log.info("✓ wrote edges: %s (%.1f MB)", out_edges, out_edges.stat().st_size / 1e6)
    log.info("✓ wrote nodes: %s (%.1f MB)", out_nodes, out_nodes.stat().st_size / 1e6)

    # ---------- step 7: stats ----------
    stats = {
        "version": "v1.2",
        "input": {
            "edges_v0_8": str(edges_v0_8_path),
            "edges_e7":   str(edges_e7_path),
            "node_idx_v0_8": str(node_idx_path),
        },
        "output": {
            "edges_v1_2_parquet": str(out_edges),
            "node_idx_v1_2_parquet": str(out_nodes),
            "n_edges_total": int(len(edges_v12)),
            "n_nodes_total": int(len(node_idx_new)),
            "n_new_nodes":  int(len(new_nodes_df)),
        },
        "edge_type_distribution": {
            t: int(n) for t, n in edges_v12["edge_type"].value_counts().items()
        },
        "_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_stats = log_dir / "build_kg_v1_2_stats.json"
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
    log.info("✓ stats JSON: %s", out_stats)

    log.info("=" * 60)
    log.info("✓ KG v1.2 construction complete")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
