#!/usr/bin/env python
"""
gnn_tuning_sweep.py  ─  MAJOR-4
==================================
粗调参 sweep: 对每个架构在 {lr, hidden_dim} 网格上扫,
证明 h=64 / lr=5e-4 的 MLP>GNN 排序不是欠调参假象.

Grid: lr ∈ {1e-3, 5e-4}  ×  hidden_dim ∈ {64, 128}  →  4 configs × 5 archs = 20 runs
Seed: 42 (单 seed, 目的是协议稳健性检验, 不是统计显著性)
Modal: M11 (saturated feature condition)

Run:
  cd ~/cyq/thesis_project/scripts/p7_gnn

  THESIS_PROJECT_ROOT=~/cyq/thesis_project \
  ~/cyq/thesis_env/.venv/bin/python \
      ~/cyq/peerj_experiments/gnn_tuning_sweep.py \
      --label-col fraud_v08_strict \
      [--seed 42] [--modal M11] [--epochs 100]

输出:
  $THESIS_PROJECT_ROOT/data/interim/p7_gnn_results/
      tuning_sweep_{label}_{modal}_seed{seed}.json  ← 全 grid 结果
      tuning_sweep_{label}_{modal}_seed{seed}_table.csv  ← 便于粘贴进论文

NOTE:
  - 原始 canonical run 用的是 lr=5e-4, hidden_dim=64 (per_seed_breakdown 里 lr 列=0.0005).
  - argparse default 是 1e-3; canonical 命令行显式传了 --lr 5e-4.
  - 这个 sweep 把两个 lr 都跑进去, 明确展示 canonical 协议的位置.
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import numpy as np

os.environ.setdefault(
    "THESIS_PROJECT_ROOT",
    str(Path.home() / "cyq" / "thesis_project"),
)

import torch

SCRIPT_DIR = Path(os.environ["THESIS_PROJECT_ROOT"]) / "scripts" / "p7_gnn"
sys.path.insert(0, str(SCRIPT_DIR))

from gnn_baseline_common import (
    setup_logger, evaluate_predictions,
    load_edge_index, load_node_index,
    SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
    RESULTS_DIR,
)
from gnn_baseline_v1_1 import (
    load_node_features_v11,
    prepare_yearly_data,
    train_and_eval,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid
# ─────────────────────────────────────────────────────────────────────────────
LR_GRID         = [1e-3, 5e-4]
HIDDEN_DIM_GRID = [64, 128]
ARCHS           = ["MLP", "GCN", "GAT", "SAGE", "RGCN"]


def run_config(model_name, modal, yearly_data, lr, hidden_dim, seed,
               args_base, logger, device) -> dict:
    """一个 (arch, lr, hidden_dim) 配置, 返回 test_auc 等."""
    import types
    cfg = types.SimpleNamespace(
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=args_base.epochs,
        label_col=args_base.label_col,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    r = train_and_eval(
        model_name, modal, yearly_data,
        SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
        cfg, logger, device,
    )
    m = r.get("metrics") or {}
    return {
        "model":      model_name,
        "modal":      modal,
        "lr":         lr,
        "hidden_dim": hidden_dim,
        "seed":       seed,
        "label_col":  args_base.label_col,
        "best_epoch": r.get("best_epoch", -1),
        "val_auc":    r.get("val_auc", 0.0),
        "test_auc":   m.get("auc",       0.0),
        "ap":         m.get("ap",        0.0),
        "f1_best":    m.get("f1_best",   0.0),
        "canonical":  (abs(lr - 5e-4) < 1e-9 and hidden_dim == 64),
    }


def print_results_table(rows, logger):
    """打印 model × (lr, hidden_dim) 的 test_auc 表."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pivot = df.pivot_table(
        index="model",
        columns=["lr", "hidden_dim"],
        values="test_auc",
        aggfunc="first",
    ).round(4)
    # 标记 canonical 配置
    logger.info("\n── test_auc sweep table (modal=%s, seed=%d) ──\n%s",
                df["modal"].iloc[0], df["seed"].iloc[0],
                pivot.to_string())

    # 找每行最高值
    logger.info("\n── best config per architecture ──")
    for model in ARCHS:
        sub = df[df.model == model]
        if sub.empty:
            continue
        best = sub.loc[sub.test_auc.idxmax()]
        can  = sub[((sub.lr - 5e-4).abs() < 1e-9) & (sub.hidden_dim == 64)]
        can_auc = can["test_auc"].values[0] if not can.empty else float("nan")
        logger.info(
            "  %s: canonical(h=64,lr=5e-4)=%.4f  best=%.4f"
            " @(h=%d,lr=%.1e)  delta=+%.4f",
            model, can_auc, best.test_auc,
            int(best.hidden_dim), best.lr,
            best.test_auc - can_auc,
        )

    # 关键问题: 任何配置下 GNN > MLP?
    logger.info("\n── Does any GNN beat MLP under any config? ──")
    for lr in LR_GRID:
        for h in HIDDEN_DIM_GRID:
            sub = df[(df.lr - lr).abs() < 1e-9 & (df.hidden_dim == h)]
            if sub.empty:
                continue
            mlp_auc = sub[sub.model == "MLP"]["test_auc"].values
            if not mlp_auc.size:
                continue
            mlp_v = mlp_auc[0]
            for gnn in ["GCN", "GAT", "SAGE", "RGCN"]:
                gnn_row = sub[sub.model == gnn]["test_auc"].values
                if not gnn_row.size:
                    continue
                gnn_v = gnn_row[0]
                if gnn_v > mlp_v:
                    logger.info(
                        "  YES: %s(%.4f) > MLP(%.4f) @ lr=%.1e h=%d  Δ=+%.4f",
                        gnn, gnn_v, mlp_v, lr, h, gnn_v - mlp_v,
                    )
                else:
                    logger.info(
                        "  no : MLP(%.4f) >= %s(%.4f) @ lr=%.1e h=%d",
                        mlp_v, gnn, gnn_v, lr, h,
                    )


def main() -> int:
    p = argparse.ArgumentParser(description="MAJOR-4: coarse GNN tuning sweep")
    p.add_argument("--models",    nargs="+", default=ARCHS)
    p.add_argument("--modal",     nargs="+", dest="modals", default=["M11"])
    p.add_argument("--label-col", type=str,   default="fraud_v08_strict")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--lr-grid",   nargs="+", type=float, default=LR_GRID)
    p.add_argument("--hd-grid",   nargs="+", type=int,   default=HIDDEN_DIM_GRID)
    p.add_argument("--device",    type=str,   default="auto")
    args = p.parse_args()

    logger = setup_logger("tuning_sweep")
    device = (
        torch.device("cuda")
        if (args.device == "auto" and torch.cuda.is_available())
        else torch.device(args.device if args.device != "auto" else "cpu")
    )
    logger.info("Device: %s | seed=%d | Modals: %s", device, args.seed, args.modals)
    logger.info("LR grid: %s | HiddenDim grid: %s", args.lr_grid, args.hd_grid)

    n_configs = len(args.lr_grid) * len(args.hd_grid) * len(args.models) * len(args.modals)
    logger.info("Total configs to run: %d", n_configs)

    logger.info("加载数据 ...")
    features_df = load_node_features_v11()
    edges_df    = load_edge_index()
    node_idx_df = load_node_index()
    logger.info("  features: %d × %d | edges: %d | nodes: %d",
                *features_df.shape, len(edges_df), len(node_idx_df))

    if args.label_col not in features_df.columns:
        logger.error("--label-col=%s 不在 features 列中!", args.label_col)
        return 2

    all_rows = []
    yearly_cache: dict = {}

    for modal in args.modals:
        if modal not in yearly_cache:
            logger.info("构建 modal=%s 年份图 ...", modal)
            yearly_cache[modal] = prepare_yearly_data(
                features_df, node_idx_df, edges_df,
                modal, args.label_col, logger,
            )

        for lr in args.lr_grid:
            for hidden_dim in args.hd_grid:
                for model_name in args.models:
                    tag = ("★ canonical" if (abs(lr - 5e-4) < 1e-9
                                             and hidden_dim == 64) else "")
                    logger.info("── %s × modal=%s lr=%.1e h=%d %s ──",
                                model_name, modal, lr, hidden_dim, tag)
                    row = run_config(
                        model_name, modal, yearly_cache[modal],
                        lr, hidden_dim, args.seed, args, logger, device,
                    )
                    all_rows.append(row)
                    logger.info("  test_auc=%.4f  val_auc=%.4f  best_epoch=%d",
                                row["test_auc"], row["val_auc"], row["best_epoch"])

    # ── 结果输出 ─────────────────────────────────────────────────────────────
    import pandas as pd
    modal_str = "-".join(args.modals)
    out_json = RESULTS_DIR / (
        f"tuning_sweep_{args.label_col}_{modal_str}_seed{args.seed}.json"
    )
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"rows": all_rows, "args": vars(args)},
                  fh, ensure_ascii=False, indent=2)
    logger.info("✓ Full results → %s", out_json)

    out_csv = out_json.with_suffix(".csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    logger.info("✓ CSV table → %s", out_csv)

    # 打印分析表
    print_results_table(all_rows, logger)

    logger.info("ALL DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
