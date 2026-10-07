"""
gnn_baseline_v1_1.py
=====================

Chapter 7 §7.3 GNN baseline 实验主脚本 (v1.1 数据).

相对 gnn_baseline_v0_1.py 的扩展:
  1. 数据源切换: node_features_v0_8.parquet → node_features_v1_1.parquet
     (109 → 129 维特征, 1 → 3 列 fraud)
  2. 模态扩展: G6/G7/G8 (3 个) → M1-M11 (11 个), 兼容旧 G6/G7/G8 别名
  3. 标签可切: --label-col 支持 fraud_v07 / fraud_v08_strict (默认) / fraud_v08_loose
  4. 实验矩阵: 38 个实验 (主表 15 + ablation 12 + robust 9 + 烟雾 2)

时序切分 (与 Chapter 6 设计 A 完全一致):
  train: 2010-2018  (24,527 firm-year)
  val:   2019-2020  ( 7,764 firm-year, early stopping)
  test:  2021-2024  (19,384 firm-year)

5 个模型架构 (沿用 v0_1):
  GCN / GAT / SAGE / RGCN / MLP

边类型 (5 类, 沿用 v0.7 KG):
  E1_HOLDS_BY / E2_FLOAT_HELD_BY / E3_HAS_MANAGER / E4_CO_HELD / E5_CO_MGR
  (E6 RPT_Repaco / E7 RPT_Operation 留给后续 §7.4 v1_2 脚本)

用法:
  # 1. 烟雾测试 (5 min, 验证管道):
  uv run --project ~/code python scripts/p7_gnn/gnn_baseline_v1_1.py \\
      --modal M11 --models MLP --epochs 10

  # 2. 主表 (Chapter 7 §7.3 主结果, ~3.5 h on GPU):
  uv run --project ~/code python scripts/p7_gnn/gnn_baseline_v1_1.py \\
      --modal M5 M10 M11 \\
      --models MLP GCN GAT SAGE RGCN \\
      --epochs 100 --label-col fraud_v08_strict

  # 3. Ablation (单模态贡献, ~2.5 h):
  uv run --project ~/code python scripts/p7_gnn/gnn_baseline_v1_1.py \\
      --modal M6 M7 M8 M9 \\
      --models MLP GCN RGCN \\
      --epochs 100 --label-col fraud_v08_strict

  # 4. Robustness (§7.5 三标签对比, ~2 h):
  for L in fraud_v07 fraud_v08_strict fraud_v08_loose; do
      uv run --project ~/code python scripts/p7_gnn/gnn_baseline_v1_1.py \\
          --modal M11 --models MLP GCN RGCN \\
          --epochs 100 --label-col $L --task-suffix _${L}
  done
"""
from __future__ import annotations

import os  # for seed override
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from gnn_baseline_common import (  # noqa: E402
    KG_DIR, RESULTS_DIR,
    SPLIT_TEST_YEARS, SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS,
    delong_test, evaluate_predictions,
    load_edge_index, load_node_index,
    setup_logger, write_results,
)


# ============================================================================
# 模态定义 (M1-M11 + 旧 G6/G7/G8 兼容)
# ============================================================================

MODAL_DEFINITIONS = {
    # --- Chapter 6 (M1-M5) ---
    "M1": ("feat_",),                                                     # 17  insider only
    "M2": ("fin_", "fini_"),                                              # 87  finance only
    "M3": ("feat_", "fin_"),                                              # 35
    "M4": ("feat_", "fini_"),                                             # 86
    "M5": ("feat_", "fin_", "fini_"),                                     # 104 = v0.7 全
    # --- Chapter 7 §7.3 ablation (单模态贡献) ---
    "M6": ("feat_", "fin_", "fini_", "audit_"),                           # 109
    "M7": ("feat_", "fin_", "fini_", "pld_"),                             # 110
    "M8": ("feat_", "fin_", "fini_", "ctrl_"),                            # 111
    "M9": ("feat_", "fin_", "fini_", "rpt_"),                             # 111
    # --- Chapter 7 §7.3 主结果 ---
    "M10": ("feat_", "fin_", "fini_", "audit_", "pld_", "ctrl_"),         # 122 = v1.0
    "M11": ("feat_", "fin_", "fini_", "audit_",
             "pld_", "ctrl_", "rpt_"),                                    # 129 = v1.1 ★
    # --- 兼容 v0_1 老命名 ---
    "G6": None,           # 仅图结构 (无特征)
    "G7": ("fin_", "fini_"),
    "G8": ("feat_", "fin_", "fini_"),
}

ALL_YEARS = SPLIT_TRAIN_YEARS + SPLIT_VAL_YEARS + SPLIT_TEST_YEARS  # 2010-2024
DEFAULT_LABEL = "fraud_v08_strict"
SUPPORTED_LABELS = ["fraud_v07", "fraud_v08_strict", "fraud_v08_loose", "fraud"]


# ============================================================================
# 数据加载
# ============================================================================

def load_node_features_v11(features_path: Path = None) -> pd.DataFrame:
    """加载 node_features_v1_1.parquet (51,675 行, ~140 列)."""
    if features_path is None:
        features_path = KG_DIR / "node_features_v1_1.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"node_features_v1_1.parquet 不存在: {features_path}\n"
            f"请先运行 prepare_node_features_v1_1.py 生成此文件")
    return pd.read_parquet(features_path)


# ============================================================================
# 模态特征构造
# ============================================================================

def select_modal_features(features_df: pd.DataFrame, modal: str) -> tuple:
    """按模态选择特征列, 返回 (feat_array, col_names)."""
    cfg = MODAL_DEFINITIONS.get(modal)
    if cfg is None:  # G6 = 仅图结构
        feat = np.ones((len(features_df), 1), dtype=np.float32)
        return feat, ["_dummy_const"]
    cols = [c for c in features_df.columns
            if any(c.startswith(p) for p in cfg)]
    cols = sorted(cols)  # 保证稳定顺序
    if not cols:
        raise ValueError(f"模态 {modal} 没有匹配到任何列, prefixes={cfg}")
    feat = features_df[cols].fillna(0).astype(np.float32).values
    return feat, cols


def fit_train_scaler(features_df: pd.DataFrame, modal: str):
    """仅基于 train 期 fit 标准化, 防止数据泄露."""
    if MODAL_DEFINITIONS.get(modal) is None:
        return None, None
    train_df = features_df[features_df["year"].isin(SPLIT_TRAIN_YEARS)]
    train_feat, _ = select_modal_features(train_df, modal)
    mu = train_feat.mean(axis=0, keepdims=True)
    sd = train_feat.std(axis=0, keepdims=True) + 1e-6
    return mu, sd


def standardize_features(feat: np.ndarray, mu, sd) -> np.ndarray:
    return feat if mu is None else (feat - mu) / sd


# ============================================================================
# 单年度 PyG Data 构造
# ============================================================================

def prepare_pyg_data_for_year(year, features_df, edges_df,
                                company_idx_map, n_total, modal, mu, sd,
                                edge_type_map, label_col, logger):
    """单年度 PyG Data. 标签从 label_col 列取."""
    from torch_geometric.data import Data

    feats_year = features_df[features_df["year"] == year].copy()
    if feats_year.empty:
        return None
    feat_array, _ = select_modal_features(feats_year, modal)
    feat_array = standardize_features(feat_array, mu, sd)
    feat_dim = feat_array.shape[1]

    X = np.zeros((n_total, feat_dim), dtype=np.float32)
    valid_mask = np.zeros(n_total, dtype=bool)
    y_array = np.zeros(n_total, dtype=np.int64)

    feats_year = feats_year.reset_index(drop=True)
    label_vals = feats_year[label_col].astype(int).values
    for i, ts in enumerate(feats_year["ts_code"].values):
        ni = company_idx_map.get(ts)
        if ni is not None:
            X[ni] = feat_array[i]
            valid_mask[ni] = True
            y_array[ni] = int(label_vals[i])

    edges_year = edges_df[edges_df["year"] == year]
    src = edges_year["src_idx"].values.astype(np.int64)
    dst = edges_year["dst_idx"].values.astype(np.int64)
    edge_type_str = edges_year["edge_type"].values
    edge_type = np.array([edge_type_map.get(e, 0) for e in edge_type_str],
                          dtype=np.int64)
    # 双向化
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ])
    edge_type_full = np.concatenate([edge_type, edge_type])

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_type=torch.tensor(edge_type_full, dtype=torch.long),
        y=torch.tensor(y_array, dtype=torch.long),
        valid_mask=torch.tensor(valid_mask, dtype=torch.bool),
    )
    data.year = year
    n_pos = int(y_array[valid_mask].sum())
    n_valid = int(valid_mask.sum())
    logger.info("  year=%d, modal=%s: x=%s, edges=%d, valid=%d, pos=%d (%.2f%%)",
                year, modal, X.shape, edge_index.shape[1],
                n_valid, n_pos, n_pos / max(n_valid, 1) * 100)
    return data


def prepare_yearly_data(features_df, node_idx_df, edges_df,
                         modal, label_col, logger):
    """一次性构造 15 年 PyG Data, 返回 dict[year -> Data]."""
    company_idx_map = dict(zip(
        node_idx_df.loc[node_idx_df["node_type"] == "company", "ts_code"],
        node_idx_df.loc[node_idx_df["node_type"] == "company", "node_idx"],
    ))
    n_total = len(node_idx_df)
    edge_type_map = {"E1_HOLDS_BY": 0, "E2_FLOAT_HELD_BY": 1,
                     "E3_HAS_MANAGER": 2, "E4_CO_HELD": 3, "E5_CO_MGR": 4}
    mu, sd = fit_train_scaler(features_df, modal)
    yearly_data = {}
    for year in ALL_YEARS:
        d = prepare_pyg_data_for_year(year, features_df, edges_df,
                                        company_idx_map, n_total,
                                        modal, mu, sd, edge_type_map,
                                        label_col, logger)
        if d is not None:
            yearly_data[year] = d
    return yearly_data


# ============================================================================
# GNN 模型 (沿用 v0_1)
# ============================================================================

def make_gnn_model(model_name, in_dim, hidden_dim, n_relations=5):
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GCNConv(in_dim, hidden_dim)
            self.c2 = GCNConv(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x, edge_index, **kw):
            x = F.relu(self.c1(x, edge_index))
            x = F.dropout(x, 0.3, training=self.training)
            x = F.relu(self.c2(x, edge_index))
            return self.head(x).squeeze(-1)

    class GAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GATConv(in_dim, hidden_dim, heads=4, concat=True, dropout=0.3)
            self.c2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False,
                                dropout=0.3)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x, edge_index, **kw):
            x = F.elu(self.c1(x, edge_index))
            x = F.elu(self.c2(x, edge_index))
            return self.head(x).squeeze(-1)

    class SAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = SAGEConv(in_dim, hidden_dim)
            self.c2 = SAGEConv(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x, edge_index, **kw):
            x = F.relu(self.c1(x, edge_index))
            x = F.dropout(x, 0.3, training=self.training)
            x = F.relu(self.c2(x, edge_index))
            return self.head(x).squeeze(-1)

    class RGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = RGCNConv(in_dim, hidden_dim, num_relations=n_relations)
            self.c2 = RGCNConv(hidden_dim, hidden_dim, num_relations=n_relations)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x, edge_index, edge_type=None, **kw):
            x = F.relu(self.c1(x, edge_index, edge_type))
            x = F.dropout(x, 0.3, training=self.training)
            x = F.relu(self.c2(x, edge_index, edge_type))
            return self.head(x).squeeze(-1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(in_dim, hidden_dim)
            self.f2 = nn.Linear(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x, edge_index=None, **kw):
            x = F.relu(self.f1(x))
            x = F.dropout(x, 0.3, training=self.training)
            x = F.relu(self.f2(x))
            return self.head(x).squeeze(-1)

    return {"GCN": GCN, "GAT": GAT, "SAGE": SAGE,
            "RGCN": RGCN, "MLP": MLP}[model_name]()


# ============================================================================
# 训练 + 评估循环
# ============================================================================

def forward_and_collect(model, data, device, scoring=False):
    data = data.to(device, non_blocking=True)
    logits = model(data.x, data.edge_index, edge_type=data.edge_type)
    valid = data.valid_mask
    out = torch.sigmoid(logits[valid]) if scoring else logits[valid]
    y = data.y[valid]
    return out, y


def train_and_eval(model_name, modal, yearly_data,
                    train_years, val_years, test_years,
                    args, logger, device):
    logger.info("=" * 60)
    logger.info("训练: model=%s, modal=%s, label=%s",
                model_name, modal, args.label_col)
    logger.info("=" * 60)

    sample_data = next(iter(yearly_data.values()))
    in_dim = sample_data.x.shape[1]
    logger.info("  in_dim=%d, hidden_dim=%d", in_dim, args.hidden_dim)

    model = make_gnn_model(model_name, in_dim, args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_pos = train_neg = 0
    for y in train_years:
        if y in yearly_data:
            d = yearly_data[y]
            tp = int(d.y[d.valid_mask].sum().item())
            tv = int(d.valid_mask.sum().item())
            train_pos += tp
            train_neg += tv - tp
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("  train pos=%d neg=%d pos_weight=%.2f",
                train_pos, train_neg, train_neg / max(train_pos, 1))

    best_val_auc = -1
    best_test_metrics = None
    best_test_score = None
    best_test_label = None
    best_epoch = -1

    for epoch in range(args.epochs):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for y in train_years:
            if y not in yearly_data:
                continue
            d = yearly_data[y]
            opt.zero_grad()
            logits, y_true = forward_and_collect(model, d, device, scoring=False)
            if y_true.numel() == 0:
                continue
            loss = loss_fn(logits, y_true.float())
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
            n_batches += 1

        # ---- VAL + TEST (every 5 epochs) ----
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_scores, val_labels = [], []
                for y in val_years:
                    if y not in yearly_data:
                        continue
                    sc, yt = forward_and_collect(model, yearly_data[y],
                                                   device, scoring=True)
                    val_scores.append(sc.cpu().numpy())
                    val_labels.append(yt.cpu().numpy())
                val_score = np.concatenate(val_scores) if val_scores else np.array([])
                val_label = np.concatenate(val_labels) if val_labels else np.array([])
                val_metrics = (evaluate_predictions(val_label, val_score)
                                if len(val_score) > 0 else {"auc": 0})

                test_scores, test_labels = [], []
                for y in test_years:
                    if y not in yearly_data:
                        continue
                    sc, yt = forward_and_collect(model, yearly_data[y],
                                                   device, scoring=True)
                    test_scores.append(sc.cpu().numpy())
                    test_labels.append(yt.cpu().numpy())
                test_score = np.concatenate(test_scores) if test_scores else np.array([])
                test_label = np.concatenate(test_labels) if test_labels else np.array([])
                test_metrics = (evaluate_predictions(test_label, test_score)
                                 if len(test_score) > 0 else {"auc": 0})

            avg_loss = train_loss_sum / max(n_batches, 1)
            logger.info("  epoch %3d: loss=%.4f val_auc=%.4f test_auc=%.4f test_f1=%.4f",
                        epoch + 1, avg_loss, val_metrics["auc"],
                        test_metrics["auc"], test_metrics.get("f1_best", 0))

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_test_metrics = test_metrics
                best_test_score = test_score.copy()
                best_test_label = test_label.copy()
                best_epoch = epoch + 1

    logger.info("  >>> BEST @ epoch %d: val_auc=%.4f, test_auc=%.4f",
                best_epoch, best_val_auc,
                best_test_metrics["auc"] if best_test_metrics else 0)

    return {
        "model": model_name, "modal": modal,
        "label_col": args.label_col,
        "best_epoch": best_epoch,
        "val_auc": float(best_val_auc),
        "metrics": best_test_metrics,
        "y_score": best_test_score.tolist() if best_test_score is not None else [],
        "y_true": best_test_label.tolist() if best_test_label is not None else [],
    }


# ============================================================================
# 主流程
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modal", nargs="+", default=["M5", "M10", "M11"],
                   choices=list(MODAL_DEFINITIONS.keys()))
    p.add_argument("--models", nargs="+",
                   default=["MLP", "GCN", "GAT", "SAGE", "RGCN"],
                   choices=["MLP", "GCN", "GAT", "SAGE", "RGCN"])
    p.add_argument("--label-col", type=str, default=DEFAULT_LABEL,
                   choices=SUPPORTED_LABELS)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--task-suffix", type=str, default="",
                   help="结果 JSON 文件名后缀, 用于多次运行不覆盖")
    p.add_argument("--features-path", type=Path, default=None,
                   help="覆写 node_features_v1_1.parquet 路径")
    p.add_argument("--seed", type=int, default=42, help="random seed (override common.py default 42)")
    args = p.parse_args()

    # 重设 seed (覆盖 common.py 在 import 时的默认 42, 用于多 seed 稳健性检验)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task = "gnn_baseline_v1_1" + args.task_suffix
    logger = setup_logger(task)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA: %s, mem: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("Args: %s", vars(args))

    # ---- 加载 ----
    logger.info("加载 v1.1 KG 数据 ...")
    features_df = load_node_features_v11(args.features_path)
    edges_df = load_edge_index()
    node_idx_df = load_node_index()
    logger.info("  node_features_v1_1: %d 行 × %d 列",
                len(features_df), len(features_df.columns))
    logger.info("  global_edges:       %d 边", len(edges_df))
    logger.info("  node_index:         %d 节点", len(node_idx_df))

    # ---- 标签可用性检查 ----
    if args.label_col not in features_df.columns:
        logger.error("--label-col=%s 不在 features 列中! 可用: %s",
                     args.label_col,
                     [c for c in features_df.columns if "fraud" in c.lower()])
        return 2
    n_pos_total = int(features_df[args.label_col].sum())
    logger.info("标签 %s: 总正样本 %d / %d (%.2f%%)",
                args.label_col, n_pos_total, len(features_df),
                n_pos_total / len(features_df) * 100)

    # ---- 时序切分 sanity ----
    logger.info("时序切分 (设计 A):")
    for split_name, years in [("train", SPLIT_TRAIN_YEARS),
                                ("val", SPLIT_VAL_YEARS),
                                ("test", SPLIT_TEST_YEARS)]:
        sub = features_df[features_df["year"].isin(years)]
        n_pos = int(sub[args.label_col].sum())
        logger.info("  %s (%d-%d): %d 行, %d 正样本 (%.2f%%)",
                    split_name, years[0], years[-1],
                    len(sub), n_pos, n_pos / max(len(sub), 1) * 100)

    # ---- 主循环 ----
    results = []
    yearly_cache = {}

    for modal in args.modal:
        if modal not in yearly_cache:
            logger.info("=" * 60)
            logger.info("构建 modal=%s 的 15 年 PyG Data ...", modal)
            logger.info("=" * 60)
            yearly_cache[modal] = prepare_yearly_data(
                features_df, node_idx_df, edges_df, modal,
                args.label_col, logger)
            logger.info("  → %d 年份就绪", len(yearly_cache[modal]))

        for model_name in args.models:
            try:
                r = train_and_eval(model_name, modal, yearly_cache[modal],
                                    SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS,
                                    SPLIT_TEST_YEARS,
                                    args, logger, device)
                if r:
                    results.append(r)
            except Exception as e:
                logger.error("%s × %s 失败: %s", model_name, modal, e)
                import traceback
                traceback.print_exc()

    # ---- 汇总 ----
    summary = []
    for r in results:
        m = r["metrics"]
        if m is None:
            continue
        summary.append({
            "model": r["model"], "modal": r["modal"],
            "label": r["label_col"],
            "best_epoch": r["best_epoch"],
            "val_auc": round(r["val_auc"], 4),
            "test_auc": round(m["auc"], 4),
            "ap": round(m["ap"], 4),
            "p_at_5pct": round(m["p_at_5pct"], 4),
            "r_at_10fpr": round(m["r_at_10fpr"], 4),
            "f1_best": round(m["f1_best"], 4),
        })
    summary_df = pd.DataFrame(summary)
    logger.info("=" * 60)
    logger.info("结果汇总:\n%s", summary_df.to_string(index=False))
    logger.info("=" * 60)

    # ---- DeLong (test 集) ----
    logger.info("DeLong 检验 (test 集)...")
    delong_results = []
    by_key = {(r["model"], r["modal"]): r for r in results if r.get("metrics")}

    # 升级模态对比 (M5 → M10 → M11)
    for model_name in args.models:
        for ma, mb in [("M5", "M10"), ("M10", "M11"), ("M5", "M11")]:
            ka, kb = (model_name, ma), (model_name, mb)
            if ka in by_key and kb in by_key:
                ra, rb = by_key[ka], by_key[kb]
                if len(ra["y_true"]) != len(rb["y_true"]):
                    continue
                d = delong_test(np.array(ra["y_true"]),
                                 np.array(ra["y_score"]),
                                 np.array(rb["y_score"]))
                delong_results.append({
                    "comparison": f"{model_name}: {mb} vs {ma}", **d,
                })

    # M11 ablation: M11 vs M6/M7/M8/M9 (单模态贡献)
    for model_name in args.models:
        for ma in ["M6", "M7", "M8", "M9"]:
            kb = (model_name, "M11")
            ka = (model_name, ma)
            if ka in by_key and kb in by_key:
                ra, rb = by_key[ka], by_key[kb]
                if len(ra["y_true"]) != len(rb["y_true"]):
                    continue
                d = delong_test(np.array(ra["y_true"]),
                                 np.array(ra["y_score"]),
                                 np.array(rb["y_score"]))
                delong_results.append({
                    "comparison": f"{model_name}: M11 vs {ma}", **d,
                })

    # GNN vs MLP (同模态)
    for modal in args.modal:
        for gnn in ["GCN", "GAT", "SAGE", "RGCN"]:
            ka, kb = ("MLP", modal), (gnn, modal)
            if ka in by_key and kb in by_key:
                ra, rb = by_key[ka], by_key[kb]
                if len(ra["y_true"]) != len(rb["y_true"]):
                    continue
                d = delong_test(np.array(ra["y_true"]),
                                 np.array(ra["y_score"]),
                                 np.array(rb["y_score"]))
                delong_results.append({
                    "comparison": f"{gnn} vs MLP ({modal})", **d,
                })

    if delong_results:
        logger.info("DeLong 检验前 15 条:")
        for d in delong_results[:15]:
            logger.info("  %s: ΔAUC=%+.4f z=%+.3f p=%.4f",
                        d["comparison"], d["delta"], d["z"], d["p"])

    write_results(task, {
        "params": vars(args),
        "summary": summary,
        "delong": delong_results,
        "n_results": len(results),
    })
    logger.info("结果已写入 %s/%s.json", RESULTS_DIR, task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
