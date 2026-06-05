#!/usr/bin/env python
"""
rerun_main_persist_scores.py  ─  MAJOR-2
=========================================
重跑 MLP + 4 GNN × 5 seed at M11 (canonical v1.1 protocol),
落盘逐样本 test_score / test_y (.npz),
计算跨种子平均分数的 DeLong 检验 (pooled multi-seed DeLong).

Run (在 4090 上):
  cd ~/cyq/thesis_project/scripts/p7_gnn

  THESIS_PROJECT_ROOT=~/cyq/thesis_project \
  ~/cyq/thesis_env/.venv/bin/python \
      ~/cyq/peerj_experiments/rerun_main_persist_scores.py \
      --label-col fraud_v08_strict \
      --modal M11 \
      --lr 5e-4

  可选加 --modal M5 M10 M11 跑三模态; 默认只跑 M11 (最快).

输出:
  $THESIS_PROJECT_ROOT/data/interim/p7_gnn_results/scores_persist/
      {model}_{modal}_{label}_{seed}_scores.npz   ← y_score(float32), y_true(int8)
  $THESIS_PROJECT_ROOT/data/interim/p7_gnn_results/
      rerun_persist_{label}_{modal_str}.json       ← per-seed 汇总 (per_seed_breakdown 兼容)
      rerun_persist_{label}_{modal_str}_delong.json ← 跨种子平均分数 DeLong
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import numpy as np

# ── THESIS_PROJECT_ROOT 必须在 import common 之前设置 ──────────────────────
os.environ.setdefault(
    "THESIS_PROJECT_ROOT",
    str(Path.home() / "cyq" / "thesis_project"),
)

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SCRIPT_DIR = Path(os.environ["THESIS_PROJECT_ROOT"]) / "scripts" / "p7_gnn"
sys.path.insert(0, str(SCRIPT_DIR))

from gnn_baseline_common import (
    setup_logger, delong_test, evaluate_predictions,
    load_edge_index, load_node_index,
    SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
    RESULTS_DIR,
)
from gnn_baseline_v1_1 import (
    load_node_features_v11,
    prepare_yearly_data,
    train_and_eval,
)

SCORES_DIR = RESULTS_DIR / "scores_persist"
SCORES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def run_one_seed(seed: int, args, features_df, node_idx_df, edges_df,
                  device, logger) -> dict:
    """一个 seed 跑所有 (model, modal), 返回 {model_name: {modal: result}}."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    results: dict = {}
    yearly_cache: dict = {}

    for modal in args.modals:
        if modal not in yearly_cache:
            logger.info("── [seed=%d] 构建 modal=%s 年份图 ──", seed, modal)
            yearly_cache[modal] = prepare_yearly_data(
                features_df, node_idx_df, edges_df,
                modal, args.label_col, logger,
            )

        for model_name in args.models:
            logger.info("── [seed=%d] 训练 %s × %s ──", seed, model_name, modal)
            r = train_and_eval(
                model_name, modal, yearly_cache[modal],
                SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
                args, logger, device,
            )

            # ── 落盘逐样本分数 ───────────────────────────────────────────
            out_npz = SCORES_DIR / (
                f"{model_name}_{modal}_{args.label_col}_{seed}_scores.npz"
            )
            np.savez_compressed(
                out_npz,
                y_score=np.array(r.get("y_score", []), dtype=np.float32),
                y_true =np.array(r.get("y_true",  []), dtype=np.int8),
                seed   =np.array([seed], dtype=np.int64),
            )
            logger.info("  ✓ scores → %s  (n=%d)", out_npz.name,
                        len(r.get("y_score", [])))

            results.setdefault(model_name, {})[modal] = r

    return results


# ─────────────────────────────────────────────────────────────────────────────
def compute_pooled_delong(args, logger) -> dict:
    """
    跨种子平均分数 DeLong:
      对每个 (GNN, modal), 取 MLP 和 GNN 各 5 seed 的 test_score,
      在同一样本维度上求平均 → 用 delong_test() 做配对检验.
    方法: 平均分数 DeLong 使用全 n=19,384 样本, 统计功效高于单种子.
    """
    delong_rows: dict = {}

    for modal in args.modals:
        scores_by_model: dict[str, np.ndarray] = {}
        y_true_ref: np.ndarray | None = None
        valid_seeds = args.seeds

        for model_name in args.models:
            arrs = []
            for seed in valid_seeds:
                p = SCORES_DIR / (
                    f"{model_name}_{modal}_{args.label_col}_{seed}_scores.npz"
                )
                if not p.exists():
                    logger.warning("缺少 %s, 跳过", p.name)
                    break
                d = np.load(p)
                arrs.append(d["y_score"].astype(np.float64))
                if y_true_ref is None:
                    y_true_ref = d["y_true"].astype(np.int64)
            if len(arrs) == len(valid_seeds):
                # shape [n_seeds, n_test] → 按样本取平均
                scores_by_model[model_name] = np.stack(arrs).mean(axis=0)

        if "MLP" not in scores_by_model or y_true_ref is None:
            logger.warning("[%s] MLP 分数缺失, 跳过 pooled DeLong", modal)
            continue

        mlp_avg = scores_by_model["MLP"]
        mlp_auc = evaluate_predictions(y_true_ref, mlp_avg)["auc"]
        modal_rows: dict = {}

        for gnn in ["GCN", "GAT", "SAGE", "RGCN"]:
            if gnn not in scores_by_model:
                logger.warning("[%s] %s 分数缺失, 跳过", modal, gnn)
                continue
            gnn_avg = scores_by_model[gnn]
            gnn_auc = evaluate_predictions(y_true_ref, gnn_avg)["auc"]
            delta   = mlp_auc - gnn_auc          # positive → MLP > GNN

            res = delong_test(y_true_ref, mlp_avg, gnn_avg)
            z   = float(res.get("z", float("nan")))
            pv  = float(res["p"])
            sig = ("***" if pv < 0.001 else "**" if pv < 0.01
                   else "*" if pv < 0.05 else "n.s.")

            logger.info(
                "[%s] Pooled DeLong  MLP(%.4f) vs %s(%.4f)  "
                "Δ=%+.4f  z=%.3f  p=%.4f  %s",
                modal, mlp_auc, gnn, gnn_auc, delta, z, pv, sig,
            )
            modal_rows[gnn] = {
                "mlp_avg_auc":          float(mlp_auc),
                "gnn_avg_auc":          float(gnn_auc),
                "delta_mlp_minus_gnn":  float(delta),
                "z_stat":               z,
                "p_value":              pv,
                "sig":                  sig,
                "n_test":               int(len(y_true_ref)),
                "n_seeds":              len(valid_seeds),
                "method":               "five-seed-mean-score DeLong",
            }
        delong_rows[modal] = modal_rows

    modal_str = "-".join(args.modals)
    out = RESULTS_DIR / f"rerun_persist_{args.label_col}_{modal_str}_delong.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(delong_rows, fh, ensure_ascii=False, indent=2)
    logger.info("✓ Pooled DeLong → %s", out)
    return delong_rows


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="MAJOR-2: rerun + persist scores")
    p.add_argument("--seeds",     nargs="+", type=int,
                   default=[42, 123, 456, 789, 1024])
    p.add_argument("--models",    nargs="+",
                   default=["MLP", "GCN", "GAT", "SAGE", "RGCN"])
    p.add_argument("--modal",     "--modals", nargs="+", dest="modals",
                   default=["M11"])
    p.add_argument("--label-col", type=str,   default="fraud_v08_strict")
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--hidden-dim",type=int,   default=64)
    p.add_argument("--lr",        type=float, default=5e-4)   # canonical lr
    p.add_argument("--device",    type=str,   default="auto")
    args = p.parse_args()

    logger = setup_logger("rerun_persist_scores")
    device = (
        torch.device("cuda") if (args.device == "auto"
                                  and torch.cuda.is_available())
        else torch.device(args.device if args.device != "auto" else "cpu")
    )
    logger.info("Device: %s | Seeds: %s | Models: %s | Modals: %s",
                device, args.seeds, args.models, args.modals)
    logger.info("lr=%.1e  hidden_dim=%d  epochs=%d", args.lr,
                args.hidden_dim, args.epochs)

    logger.info("加载数据 ...")
    features_df = load_node_features_v11()
    edges_df    = load_edge_index()
    node_idx_df = load_node_index()
    logger.info("  features: %d × %d | edges: %d | nodes: %d",
                *features_df.shape, len(edges_df), len(node_idx_df))

    if args.label_col not in features_df.columns:
        logger.error("--label-col=%s 不在 features 列中!", args.label_col)
        return 2

    # ── 主循环 ───────────────────────────────────────────────────────────────
    all_results: dict = {}
    for seed in args.seeds:
        logger.info("═" * 60)
        logger.info("SEED = %d", seed)
        logger.info("═" * 60)
        all_results[seed] = run_one_seed(
            seed, args, features_df, node_idx_df, edges_df, device, logger,
        )

    # ── per-seed 汇总 JSON (per_seed_breakdown.csv 兼容格式) ─────────────────
    summary_rows = []
    for seed, models_res in all_results.items():
        for model_name, modals_res in models_res.items():
            for modal, r in modals_res.items():
                m = r.get("metrics") or {}
                summary_rows.append({
                    "seed":       seed,
                    "model":      model_name,
                    "modal":      modal,
                    "label_col":  args.label_col,
                    "hidden_dim": args.hidden_dim,
                    "lr":         args.lr,
                    "best_epoch": r.get("best_epoch", -1),
                    "val_auc":    r.get("val_auc", 0.0),
                    "test_auc":   m.get("auc",        0.0),
                    "ap":         m.get("ap",          0.0),
                    "f1_best":    m.get("f1_best",     0.0),
                    "p_at_5pct":  m.get("p_at_5pct",  0.0),
                    "r_at_10fpr": m.get("r_at_10fpr", 0.0),
                })

    modal_str = "-".join(args.modals)
    out_sum = RESULTS_DIR / f"rerun_persist_{args.label_col}_{modal_str}.json"
    with open(out_sum, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary_rows, "args": vars(args)},
                  fh, ensure_ascii=False, indent=2)
    logger.info("✓ Per-seed summary → %s", out_sum)

    # 打印 M11 汇总表
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        tbl = (df.groupby(["model", "modal"])["test_auc"]
               .agg(["mean", "std", "count"])
               .round(4))
        logger.info("\nFive-seed AUC summary:\n%s", tbl.to_string())

    # ── Pooled DeLong ────────────────────────────────────────────────────────
    if len(args.seeds) > 1:
        logger.info("计算跨种子平均分数 DeLong ...")
        compute_pooled_delong(args, logger)

    logger.info("ALL DONE.  scores at %s", SCORES_DIR)
    logger.info("summary  at %s", out_sum)
    return 0


if __name__ == "__main__":
    sys.exit(main())
