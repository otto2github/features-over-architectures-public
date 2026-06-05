#!/usr/bin/env python
"""
run_pcgnn.py  ─  MAJOR-1
==========================
在与主基准完全相同的 v1.1 协议下跑 PC-GNN (Liu et al., WWW 2021),
落盘逐样本 test_score / test_y, 输出 per-seed 汇总.

PC-GNN 核心机制 (PyG 实现):
  ChooseStep: 对每条边计算源/目的节点的余弦相似度, 以 sigmoid 门控
              聚合邻居消息 (正相似 → 全权重, 负相似 → 趋零权重),
              即"选择相似邻居"的可微近似.
  PickStep:   通过现有的类别加权 BCE loss (pos_weight) 实现正负样本平衡.

协议与主基准完全匹配:
  - 图: global_edge_index.parquet (v1.1, 5 edge types)
  - 特征: node_features_v1_1.parquet, modal=M11
  - 时序切分: 2010-2018 train / 2019-2020 val / 2021-2024 test
  - 种子: {42, 123, 456, 789, 1024}
  - hidden_dim=64, lr=5e-4, epochs=100, early-stopping patience=10 (epoch 粒度)

Run:
  cd ~/cyq/thesis_project/scripts/p7_gnn

  THESIS_PROJECT_ROOT=~/cyq/thesis_project \
  ~/cyq/thesis_env/.venv/bin/python \
      ~/cyq/peerj_experiments/run_pcgnn.py \
      --label-col fraud_v08_strict \
      --modal M11

输出:
  $THESIS_PROJECT_ROOT/data/interim/p7_gnn_results/scores_persist/
      PCGNN_{modal}_{label}_{seed}_scores.npz
  $THESIS_PROJECT_ROOT/data/interim/p7_gnn_results/
      pcgnn_{label}_{modal_str}.json   ← per-seed-breakdown 兼容格式
"""
from __future__ import annotations
import argparse, json, sys, os, math
from pathlib import Path
import numpy as np

os.environ.setdefault(
    "THESIS_PROJECT_ROOT",
    str(Path.home() / "cyq" / "thesis_project"),
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

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
)

SCORES_DIR = RESULTS_DIR / "scores_persist"
SCORES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PC-GNN (PyG)
# ─────────────────────────────────────────────────────────────────────────────

class PCGNNConv(MessagePassing):
    """
    PC-GNN ChooseStep 消息传递层.

    对每条 (src→dst) 边计算余弦相似度 sim = cos(x_src, x_dst),
    用 gate = sigmoid(sim × scale) 加权邻居消息 (scale=10 使门控接近 0/1),
    再按目标节点的权重总和归一化, 保证输出量级稳定.

    这是 PC-GNN ChooseStep 的可微近似:
      - sim > 0 → gate → 1: 相似邻居, 全权参与聚合
      - sim < 0 → gate → 0: 不相似(可能伪装)邻居, 被抑制
    """
    def __init__(self, in_dim: int, out_dim: int, gate_scale: float = 10.0):
        super().__init__(aggr="add", node_dim=0)
        self.lin_msg  = nn.Linear(in_dim, out_dim, bias=False)
        self.gate_scale = gate_scale

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                **kw) -> torch.Tensor:
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # 余弦相似度 per edge
        x_norm = F.normalize(x, p=2, dim=-1)          # [N, d]
        sim    = (x_norm[src] * x_norm[dst]).sum(-1)   # [E]

        # Soft ChooseStep gate
        gate = torch.sigmoid(sim * self.gate_scale)    # [E] ∈ (0, 1)

        # 按目标节点归一化 (使聚合 ≈ 加权平均)
        gate_sum = torch.zeros(N, dtype=gate.dtype, device=gate.device)
        gate_sum.scatter_add_(0, dst, gate)
        gate_norm = gate / (gate_sum[dst].clamp(min=1e-6))  # [E]

        return self.propagate(edge_index, x=x, gate=gate_norm)

    def message(self, x_j: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        # x_j: [E, in_dim], gate: [E]
        return gate.unsqueeze(-1) * self.lin_msg(x_j)  # [E, out_dim]


class PCGNN(nn.Module):
    """
    2-layer PC-GNN with skip connections (residual) and dropout.
    Architecture matches the other GNNs in gnn_baseline_v1_1:
      Conv1(in → h) → dropout → Conv2(h → h) → Linear(h → 1)
    """
    def __init__(self, in_dim: int, hidden_dim: int,
                 dropout: float = 0.3, gate_scale: float = 10.0):
        super().__init__()
        self.conv1  = PCGNNConv(in_dim,     hidden_dim, gate_scale)
        self.conv2  = PCGNNConv(hidden_dim, hidden_dim, gate_scale)
        self.skip1  = nn.Linear(in_dim,     hidden_dim, bias=False)
        self.skip2  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head   = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                **kw) -> torch.Tensor:
        # Layer 1 with residual
        h = F.elu(self.conv1(x, edge_index) + self.skip1(x))
        h = F.dropout(h, self.dropout, training=self.training)
        # Layer 2 with residual
        h = F.elu(self.conv2(h, edge_index) + self.skip2(h))
        return self.head(h)                # [N, 1] logits


# ─────────────────────────────────────────────────────────────────────────────
# Training loop  (mirrors train_and_eval from gnn_baseline_v1_1)
# ─────────────────────────────────────────────────────────────────────────────

def _forward_and_collect(model, data, device, scoring=False):
    data   = data.to(device, non_blocking=True)
    logits = model(data.x, data.edge_index)
    valid  = data.valid_mask
    out    = torch.sigmoid(logits[valid]) if scoring else logits[valid]
    y      = data.y[valid]
    return out, y


def train_and_eval_pcgnn(modal, yearly_data, train_years, val_years,
                          test_years, args, logger, device) -> dict:
    sample = next(iter(yearly_data.values()))
    in_dim = sample.x.shape[1]
    logger.info("PC-GNN: in_dim=%d  hidden_dim=%d  lr=%.1e",
                in_dim, args.hidden_dim, args.lr)

    model = PCGNN(in_dim, args.hidden_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=args.lr, weight_decay=1e-5)

    # Compute pos_weight from training set
    train_pos = train_neg = 0
    for y in train_years:
        if y in yearly_data:
            d = yearly_data[y]
            tp = int(d.y[d.valid_mask].sum().item())
            tv = int(d.valid_mask.sum().item())
            train_pos += tp
            train_neg += tv - tp
    pos_w   = torch.tensor([train_neg / max(train_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    logger.info("  train pos=%d neg=%d pos_weight=%.2f",
                train_pos, train_neg, pos_w.item())

    best_val_auc   = -1.0
    best_metrics   = None
    best_score     = None
    best_label     = None
    best_epoch     = -1
    patience_count = 0
    patience       = 10

    for epoch in range(args.epochs):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        for y in train_years:
            if y not in yearly_data:
                continue
            d = yearly_data[y]
            opt.zero_grad()
            logits, y_true = _forward_and_collect(model, d, device, scoring=False)
            if y_true.numel() == 0:
                continue
            loss = loss_fn(logits.squeeze(-1), y_true.float())
            loss.backward()
            opt.step()

        # ── val + test every 5 epochs ───────────────────────────────────────
        if (epoch + 1) % 5 != 0 and epoch != args.epochs - 1:
            continue

        model.eval()
        with torch.no_grad():
            val_sc, val_lb = [], []
            for y in val_years:
                if y not in yearly_data:
                    continue
                sc, lb = _forward_and_collect(model, yearly_data[y],
                                               device, scoring=True)
                val_sc.append(sc.cpu().numpy())
                val_lb.append(lb.cpu().numpy())
            v_score = np.concatenate(val_sc) if val_sc else np.array([])
            v_label = np.concatenate(val_lb) if val_lb else np.array([])
            val_auc = (evaluate_predictions(v_label, v_score)["auc"]
                       if len(v_score) > 0 else 0.0)

            tst_sc, tst_lb = [], []
            for y in test_years:
                if y not in yearly_data:
                    continue
                sc, lb = _forward_and_collect(model, yearly_data[y],
                                               device, scoring=True)
                tst_sc.append(sc.cpu().numpy())
                tst_lb.append(lb.cpu().numpy())
            t_score = np.concatenate(tst_sc) if tst_sc else np.array([])
            t_label = np.concatenate(tst_lb) if tst_lb else np.array([])
            t_met   = (evaluate_predictions(t_label, t_score)
                       if len(t_score) > 0 else {"auc": 0})

        logger.info("  epoch %3d: val_auc=%.4f  test_auc=%.4f",
                    epoch + 1, val_auc, t_met.get("auc", 0))

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_metrics   = t_met
            best_score     = t_score.copy()
            best_label     = t_label.copy()
            best_epoch     = epoch + 1
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience // 5:   # patience in 5-epoch ticks
                logger.info("  early stop @ epoch %d", epoch + 1)
                break

    logger.info("  BEST @ epoch %d: val_auc=%.4f  test_auc=%.4f",
                best_epoch, best_val_auc,
                best_metrics["auc"] if best_metrics else 0)
    return {
        "model":       "PCGNN",
        "modal":       modal,
        "best_epoch":  best_epoch,
        "val_auc":     float(best_val_auc),
        "metrics":     best_metrics,
        "y_score":     best_score.tolist() if best_score is not None else [],
        "y_true":      best_label.tolist() if best_label is not None else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="MAJOR-1: PC-GNN benchmark run")
    p.add_argument("--seeds",     nargs="+", type=int,
                   default=[42, 123, 456, 789, 1024])
    p.add_argument("--modal",     nargs="+", dest="modals", default=["M11"])
    p.add_argument("--label-col", type=str,   default="fraud_v08_strict")
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--hidden-dim",type=int,   default=64)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--gate-scale",type=float, default=10.0,
                   help="ChooseStep sigmoid gate sharpness (default 10.0)")
    p.add_argument("--device",    type=str,   default="auto")
    args = p.parse_args()

    logger = setup_logger("run_pcgnn")
    device = (
        torch.device("cuda")
        if (args.device == "auto" and torch.cuda.is_available())
        else torch.device(args.device if args.device != "auto" else "cpu")
    )
    logger.info("Device: %s | Seeds: %s | Modals: %s", device,
                args.seeds, args.modals)
    logger.info("hidden_dim=%d  lr=%.1e  gate_scale=%.1f",
                args.hidden_dim, args.lr, args.gate_scale)

    logger.info("加载数据 ...")
    features_df = load_node_features_v11()
    edges_df    = load_edge_index()
    node_idx_df = load_node_index()
    logger.info("  features: %d × %d | edges: %d | nodes: %d",
                *features_df.shape, len(edges_df), len(node_idx_df))

    if args.label_col not in features_df.columns:
        logger.error("--label-col=%s 不在 features 列中!", args.label_col)
        return 2

    summary_rows = []
    yearly_cache: dict = {}

    for seed in args.seeds:
        logger.info("═" * 60)
        logger.info("SEED = %d", seed)
        logger.info("═" * 60)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for modal in args.modals:
            if modal not in yearly_cache:
                logger.info("构建 modal=%s 年份图 ...", modal)
                yearly_cache[modal] = prepare_yearly_data(
                    features_df, node_idx_df, edges_df,
                    modal, args.label_col, logger,
                )

            r = train_and_eval_pcgnn(
                modal, yearly_cache[modal],
                SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
                args, logger, device,
            )

            # 落盘分数
            out_npz = SCORES_DIR / (
                f"PCGNN_{modal}_{args.label_col}_{seed}_scores.npz"
            )
            np.savez_compressed(
                out_npz,
                y_score=np.array(r["y_score"], dtype=np.float32),
                y_true =np.array(r["y_true"],  dtype=np.int8),
                seed   =np.array([seed], dtype=np.int64),
            )
            logger.info("  ✓ scores → %s  (n=%d)", out_npz.name,
                        len(r["y_score"]))

            m = r.get("metrics") or {}
            summary_rows.append({
                "seed":       seed,
                "model":      "PCGNN",
                "modal":      modal,
                "label_col":  args.label_col,
                "hidden_dim": args.hidden_dim,
                "lr":         args.lr,
                "gate_scale": args.gate_scale,
                "best_epoch": r.get("best_epoch", -1),
                "val_auc":    r.get("val_auc", 0.0),
                "test_auc":   m.get("auc",       0.0),
                "ap":         m.get("ap",        0.0),
                "f1_best":    m.get("f1_best",   0.0),
                "p_at_5pct":  m.get("p_at_5pct",  0.0),
                "r_at_10fpr": m.get("r_at_10fpr", 0.0),
            })

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    modal_str = "-".join(args.modals)
    out_json  = RESULTS_DIR / f"pcgnn_{args.label_col}_{modal_str}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary_rows, "args": vars(args)},
                  fh, ensure_ascii=False, indent=2)
    logger.info("✓ Summary → %s", out_json)

    import pandas as pd
    df  = pd.DataFrame(summary_rows)
    tbl = (df.groupby(["model", "modal"])["test_auc"]
           .agg(["mean", "std", "count"])
           .round(4))
    logger.info("\nPC-GNN five-seed AUC summary:\n%s", tbl.to_string())

    # 与 MLP/GNN Table 2 对比 (从 scores_persist 加载 MLP 分数)
    logger.info("\n与主基准 MLP M11 对比 (若 scores_persist 中有 MLP 分数):")
    mlp_aucs = []
    for seed in args.seeds:
        p = SCORES_DIR / f"MLP_M11_{args.label_col}_{seed}_scores.npz"
        if p.exists():
            d = np.load(p)
            mlp_aucs.append(evaluate_predictions(
                d["y_true"].astype(int), d["y_score"].astype(float))["auc"])
    if mlp_aucs:
        logger.info("  MLP M11 five-seed AUC: %.4f ± %.4f",
                    np.mean(mlp_aucs), np.std(mlp_aucs, ddof=1))
        pcgnn_aucs = [r["metrics"]["auc"] for r in
                      [s for s in [{"metrics": {"auc": df[
                          (df.modal == "M11")]["test_auc"].mean()}}]]
                      if r["metrics"]]
        # Simple comparison
        pcgnn_mean = df[df.modal == "M11"]["test_auc"].mean() if not df.empty else 0
        logger.info("  PCGNN M11 five-seed AUC: %.4f ± %.4f",
                    df[df.modal == "M11"]["test_auc"].mean(),
                    df[df.modal == "M11"]["test_auc"].std())
        logger.info("  Δ(PCGNN - MLP) = %+.4f",
                    pcgnn_mean - np.mean(mlp_aucs))
    else:
        logger.info("  (先跑 rerun_main_persist_scores.py 落盘 MLP 分数后,")
        logger.info("   再重跑本脚本可自动打印对比)")

    logger.info("ALL DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
