#!/usr/bin/env python
"""
exp1_audit_ablation.py  -  MAJOR-1 (Reviewer 1)
================================================
回应审稿意见 MAJOR-1: audit-opinion 语义接近标签。
量化去掉 audit 块后,披露特征信号还剩多少。

新增模态 M11_no_audit = M11 去掉 5 个 audit_ 列 (124 维)
对 MLP + 4 GNN x 5 seed, 跑 M5 / M11_no_audit / M11 三模态

运行 (4090):
  cd ~/cyq/thesis_project/scripts/p7_gnn
  THESIS_PROJECT_ROOT=~/cyq/thesis_project \
  ~/cyq/thesis_env/.venv/bin/python \
      ~/cyq/peerj_experiments/exp1_audit_ablation.py \
      --label-col fraud_v08_strict

预计 ~50-70 分钟。表格学习器用 exp1b_audit_tabular.py。
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("THESIS_PROJECT_ROOT",
                      str(Path.home() / "cyq" / "thesis_project"))
import torch

SCRIPT_DIR = Path(os.environ["THESIS_PROJECT_ROOT"]) / "scripts" / "p7_gnn"
sys.path.insert(0, str(SCRIPT_DIR))

from gnn_baseline_common import (
    setup_logger, SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
    RESULTS_DIR, load_edge_index, load_node_index,
)
import gnn_baseline_v1_1 as base
from gnn_baseline_v1_1 import (
    load_node_features_v11, prepare_yearly_data, train_and_eval,
)

# 注入新模态: M11_no_audit = M11 去 audit_  (124 维, 无 audit)
base.MODAL_DEFINITIONS["M11_no_audit"] = (
    "feat_", "fin_", "fini_", "pld_", "ctrl_", "rpt_",
)

SEEDS  = [42, 123, 456, 789, 1024]
MODALS = ["M5", "M11_no_audit", "M11"]
MODELS = ["MLP", "GCN", "GAT", "SAGE", "RGCN"]   # baseline 里是 SAGE


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label-col", default="fraud_v08_strict")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    logger = setup_logger("audit_ablation")
    t0 = time.time()
    logger.info("=" * 64)
    logger.info("EXP1 Audit Ablation (MAJOR-1)")
    logger.info("Modals: %s | Models: %s | Seeds: %s", MODALS, MODELS, SEEDS)
    logger.info("Device: %s", device)
    logger.info("=" * 64)

    features_df = load_node_features_v11()
    edges_df    = load_edge_index()
    node_idx_df = load_node_index()
    logger.info("features=%s edges=%s nodes=%s",
                features_df.shape, edges_df.shape, node_idx_df.shape)

    results = {m: {md: [] for md in MODALS} for m in MODELS}

    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for modal in MODALS:
            logger.info("-- [seed=%d] build modal=%s --", seed, modal)
            yearly = prepare_yearly_data(
                features_df, node_idx_df, edges_df,
                modal, args.label_col, logger,
            )
            for model_name in MODELS:
                logger.info("-- train %s / %s / seed=%d --",
                            model_name, modal, seed)
                try:
                    r = train_and_eval(
                        model_name, modal, yearly,
                        SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
                        args, logger, device,
                    )
                    auc = float(r["metrics"]["auc"])
                except Exception as e:
                    logger.error("   ERROR %s/%s/seed=%d: %s",
                                 model_name, modal, seed, e)
                    auc = float("nan")
                results[model_name][modal].append(auc)
                logger.info("   AUC=%.4f", auc)

    logger.info("\n" + "=" * 64)
    logger.info("SUMMARY (five-seed mean +/- std)")
    logger.info("%-10s %-16s %-16s %-16s %-14s",
                "Model", "M5", "M11_no_audit", "M11", "audit_gain")
    summary = {}
    for model_name in MODELS:
        row = {}
        for modal in MODALS:
            arr = results[model_name][modal]
            row[modal] = {"aucs": arr,
                          "mean": float(np.nanmean(arr)),
                          "std":  float(np.nanstd(arr))}
        row["audit_gain"]    = row["M11"]["mean"] - row["M11_no_audit"]["mean"]
        row["nonaudit_gain"] = row["M11_no_audit"]["mean"] - row["M5"]["mean"]
        summary[model_name] = row
        logger.info("%-10s %.4f+/-%.4f  %.4f+/-%.4f  %.4f+/-%.4f  %+.4f",
                    model_name,
                    row["M5"]["mean"], row["M5"]["std"],
                    row["M11_no_audit"]["mean"], row["M11_no_audit"]["std"],
                    row["M11"]["mean"], row["M11"]["std"],
                    row["audit_gain"])

    ts  = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"audit_ablation_{args.label_col}_{ts}.json"
    out.write_text(json.dumps({
        "experiment": "audit_ablation_MAJOR1",
        "seeds": SEEDS, "modals": MODALS, "models": MODELS,
        "label_col": args.label_col,
        "audit_cols_removed": ["audit_is_modified", "audit_type_code",
                               "audit_fee_log", "audit_fee_to_assets",
                               "audit_firm_change_flag"],
        "results": summary,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }, indent=2, ensure_ascii=False))
    logger.info("\nSaved -> %s", out)
    logger.info("Total: %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
