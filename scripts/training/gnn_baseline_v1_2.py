"""
gnn_baseline_v1_2.py
=====================

Heterogeneous graph GNN Training (KG v1.2: E1-E5 + E7).

relative to gnn_baseline_v1_1.py  extends:
  1. KG switched: global_edge_index.parquet → global_edge_index_v1_2.parquet
                node_id_index.parquet     → node_id_index_v1_2.parquet
  2. edge_type_map extended: 5 types → 10 types (E1-E5 + E7_TRADE_FUND/GUARANTEE/COMMERCIAL/ASSET/OTHER)
  3. RGCN   num_relations automatically set to 10
  4. modelonly evaluates GCN/RGCN/MLP (other models in v1.1 verified to show no significant gain)

Main research question:RPT_Operation related-party transactionadded as KG edges, whether RGCN/GAT outperforms MLP?

Usage:
  uv run --project ${PROJECT_ROOT:-~/code/foa_project}/.venv python gnn_baseline_v1_2.py \\
      --modal M11 --models GCN GAT RGCN MLP \\
      --epochs 40 --hidden-dim 64 --lr 5e-4 \\
      --label-col fraud_v08_strict \\
      --task-suffix _v74_main
"""
from __future__ import annotations

import os  # for EDGE_PATH env override

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
    setup_logger, write_results,
)


# ============================================================================
# v1.2 KG load
# ============================================================================

def load_edge_index_v12(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = KG_DIR / "global_edge_index_v1_2.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"v1.2 KG does not exist: {path}\nplease run first build_kg_v1_2.py")
    return pd.read_parquet(path)


def load_node_index_v12(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = KG_DIR / "node_id_index_v1_2.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"v1.2 node indexdoes not exist: {path}\nplease run first build_kg_v1_2.py")
    return pd.read_parquet(path)


def load_node_features_v11(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = KG_DIR / "node_features_v1_1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"node_features_v1_1.parquet does not exist: {path}")
    return pd.read_parquet(path)


# ============================================================================
# Modality definitions (consistent with v1_1)
# ============================================================================

MODAL_DEFINITIONS = {
    "M5": ("feat_", "fin_", "fini_"),
    "M10": ("feat_", "fin_", "fini_", "audit_", "pld_", "ctrl_"),
    "M11": ("feat_", "fin_", "fini_", "audit_", "pld_", "ctrl_", "rpt_"),
    "G6": None,  # graph structure only
}

ALL_YEARS = SPLIT_TRAIN_YEARS + SPLIT_VAL_YEARS + SPLIT_TEST_YEARS
SUPPORTED_LABELS = ["fraud_v07", "fraud_v08_strict", "fraud_v08_loose"]


def select_modal_features(features_df, modal):
    cfg = MODAL_DEFINITIONS.get(modal)
    if cfg is None:
        feat = np.ones((len(features_df), 1), dtype=np.float32)
        return feat, ["_dummy_const"]
    cols = sorted([c for c in features_df.columns
                    if any(c.startswith(p) for p in cfg)])
    if not cols:
        raise ValueError(f"modality {modal} no matching features")
    feat = features_df[cols].fillna(0).astype(np.float32).values
    return feat, cols


def fit_train_scaler(features_df, modal):
    if MODAL_DEFINITIONS.get(modal) is None:
        return None, None
    train_df = features_df[features_df["year"].isin(SPLIT_TRAIN_YEARS)]
    train_feat, _ = select_modal_features(train_df, modal)
    mu = train_feat.mean(axis=0, keepdims=True)
    sd = train_feat.std(axis=0, keepdims=True) + 1e-6
    return mu, sd


def standardize(feat, mu, sd):
    return feat if mu is None else (feat - mu) / sd


# ============================================================================
# Per-year graph
# ============================================================================

def prepare_pyg_data_for_year(year, features_df, edges_df,
                                company_idx_map, n_total, modal, mu, sd,
                                edge_type_map, label_col, logger):
    from torch_geometric.data import Data

    feats_year = features_df[features_df["year"] == year].copy()
    if feats_year.empty:
        return None
    feat_array, _ = select_modal_features(feats_year, modal)
    feat_array = standardize(feat_array, mu, sd)
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
    logger.info("  year=%d: x=%s, edges=%d, valid=%d, pos=%d (%.2f%%)",
                year, X.shape, edge_index.shape[1],
                n_valid, n_pos, n_pos / max(n_valid, 1) * 100)
    return data


def prepare_yearly_data(features_df, node_idx_df, edges_df,
                         modal, label_col, logger):
    company_idx_map = dict(zip(
        node_idx_df.loc[node_idx_df["node_type"] == "company", "ts_code"],
        node_idx_df.loc[node_idx_df["node_type"] == "company", "node_idx"],
    ))
    n_total = len(node_idx_df)

    # v1.2 edge_type_map: auto-extract all edge_type values from data
    unique_edge_types = sorted(edges_df["edge_type"].unique())
    edge_type_map = {t: i for i, t in enumerate(unique_edge_types)}
    logger.info("  v1.2 edge_type_map (%d types): %s",
                len(edge_type_map), edge_type_map)

    mu, sd = fit_train_scaler(features_df, modal)
    yearly = {}
    for year in ALL_YEARS:
        d = prepare_pyg_data_for_year(year, features_df, edges_df,
                                        company_idx_map, n_total,
                                        modal, mu, sd, edge_type_map,
                                        label_col, logger)
        if d is not None:
            yearly[year] = d
    return yearly, len(edge_type_map)


# ============================================================================
# GNN (n_relations dynamic)
# ============================================================================

def make_gnn_model(model_name, in_dim, hidden_dim, n_relations):
    from torch_geometric.nn import GCNConv, GATConv, RGCNConv

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

    return {"GCN": GCN, "GAT": GAT, "RGCN": RGCN, "MLP": MLP}[model_name]()


def forward_collect(model, data, device, scoring=False):
    data = data.to(device, non_blocking=True)
    logits = model(data.x, data.edge_index, edge_type=data.edge_type)
    valid = data.valid_mask
    out = torch.sigmoid(logits[valid]) if scoring else logits[valid]
    return out, data.y[valid]


def train_and_eval(model_name, modal, yearly, n_relations,
                    train_yrs, val_yrs, test_yrs,
                    args, logger, device):
    logger.info("=" * 60)
    logger.info("Training: %s × %s × %s", model_name, modal, args.label_col)
    logger.info("=" * 60)

    sample = next(iter(yearly.values()))
    in_dim = sample.x.shape[1]
    logger.info("  in_dim=%d, hidden=%d, n_relations=%d",
                in_dim, args.hidden_dim, n_relations)

    model = make_gnn_model(model_name, in_dim, args.hidden_dim, n_relations).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_pos = train_neg = 0
    for y in train_yrs:
        if y in yearly:
            d = yearly[y]
            tp = int(d.y[d.valid_mask].sum().item())
            tv = int(d.valid_mask.sum().item())
            train_pos += tp
            train_neg += tv - tp
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1
    best_metrics = None
    best_score = None
    best_label = None
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        nb = 0
        for y in train_yrs:
            if y not in yearly:
                continue
            d = yearly[y]
            opt.zero_grad()
            logits, yt = forward_collect(model, d, device, scoring=False)
            if yt.numel() == 0:
                continue
            loss = loss_fn(logits, yt.float())
            loss.backward()
            opt.step()
            train_loss += loss.item()
            nb += 1

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vs, vl = [], []
                for y in val_yrs:
                    if y in yearly:
                        s, t = forward_collect(model, yearly[y], device, scoring=True)
                        vs.append(s.cpu().numpy()); vl.append(t.cpu().numpy())
                val_score = np.concatenate(vs) if vs else np.array([])
                val_label = np.concatenate(vl) if vl else np.array([])
                vm = evaluate_predictions(val_label, val_score) if len(val_score) > 0 else {"auc": 0}

                ts, tl = [], []
                for y in test_yrs:
                    if y in yearly:
                        s, t = forward_collect(model, yearly[y], device, scoring=True)
                        ts.append(s.cpu().numpy()); tl.append(t.cpu().numpy())
                test_score = np.concatenate(ts) if ts else np.array([])
                test_label = np.concatenate(tl) if tl else np.array([])
                tm = evaluate_predictions(test_label, test_score) if len(test_score) > 0 else {"auc": 0}

            logger.info("  ep %3d: loss=%.4f val_auc=%.4f test_auc=%.4f f1=%.4f",
                        epoch+1, train_loss/max(nb,1),
                        vm["auc"], tm["auc"], tm.get("f1_best", 0))

            if vm["auc"] > best_val_auc:
                best_val_auc = vm["auc"]
                best_metrics = tm
                best_score = test_score.copy()
                best_label = test_label.copy()
                best_epoch = epoch + 1

    logger.info("  >>> BEST @ ep %d: val=%.4f, test=%.4f",
                best_epoch, best_val_auc,
                best_metrics["auc"] if best_metrics else 0)

    return {
        "model": model_name, "modal": modal,
        "label_col": args.label_col,
        "best_epoch": best_epoch,
        "val_auc": float(best_val_auc),
        "metrics": best_metrics,
        "y_score": best_score.tolist() if best_score is not None else [],
        "y_true": best_label.tolist() if best_label is not None else [],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--modal", nargs="+", default=["M5", "M11"],
                   choices=list(MODAL_DEFINITIONS.keys()))
    p.add_argument("--models", nargs="+",
                   default=["MLP", "GCN", "GAT", "RGCN"],
                   choices=["MLP", "GCN", "GAT", "RGCN"])
    p.add_argument("--label-col", type=str, default="fraud_v08_strict",
                   choices=SUPPORTED_LABELS)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--task-suffix", type=str, default="")
    p.add_argument("--seed", type=int, default=42, help="random seed (override common.py default 42)")
    args = p.parse_args()

    # Reset seed (overrides the default 42 set in common.py at import time; used for multi-seed robustness)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task = "gnn_baseline_v1_2" + args.task_suffix
    logger = setup_logger(task)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                            else args.device if args.device != "auto" else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA: %s, mem: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("Args: %s", vars(args))

    logger.info("load v1.2 KG ...")
    features_df = load_node_features_v11()
    _edge_path_env = os.environ.get('EDGE_PATH')
    if _edge_path_env:
        from pathlib import Path as _P
        edges_df = load_edge_index_v12(path=_P(_edge_path_env))
        logger.info('  [EDGE_PATH override] using: %s', _edge_path_env)
    else:
        edges_df = load_edge_index_v12()
    node_idx_df = load_node_index_v12()
    logger.info("  features: %d × %d", len(features_df), len(features_df.columns))
    logger.info("  edges:    %d", len(edges_df))
    logger.info("  nodes:    %d", len(node_idx_df))
    logger.info("  edge_type distribution:")
    for t, n in edges_df["edge_type"].value_counts().items():
        logger.info("    %-25s %d", t, n)

    if args.label_col not in features_df.columns:
        logger.error("--label-col=%s does not exist", args.label_col)
        return 2

    results = []
    yearly_cache = {}
    n_relations_cache = None

    for modal in args.modal:
        if modal not in yearly_cache:
            logger.info("=" * 60)
            logger.info("Construct 15-year PyG Data for %s ...", modal)
            logger.info("=" * 60)
            y, nr = prepare_yearly_data(features_df, node_idx_df, edges_df,
                                          modal, args.label_col, logger)
            yearly_cache[modal] = y
            n_relations_cache = nr
            logger.info("  → %d years ready, n_relations=%d", len(y), nr)

        for model_name in args.models:
            try:
                r = train_and_eval(model_name, modal, yearly_cache[modal],
                                    n_relations_cache,
                                    SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS,
                                    SPLIT_TEST_YEARS, args, logger, device)
                if r:
                    results.append(r)
            except Exception as e:
                logger.error("%s × %s failed: %s", model_name, modal, e)
                import traceback; traceback.print_exc()

    summary = []
    for r in results:
        m = r["metrics"]
        if not m:
            continue
        summary.append({
            "model": r["model"], "modal": r["modal"], "label": r["label_col"],
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
    logger.info("Results summary:\n%s", summary_df.to_string(index=False))

    # DeLong: v1.2 vs MLP, each model under each modality
    logger.info("DeLong test:")
    delong_results = []
    by_key = {(r["model"], r["modal"]): r for r in results if r.get("metrics")}
    for modal in args.modal:
        for gnn in ["GCN", "GAT", "RGCN"]:
            ka, kb = ("MLP", modal), (gnn, modal)
            if ka in by_key and kb in by_key:
                ra, rb = by_key[ka], by_key[kb]
                if len(ra["y_true"]) != len(rb["y_true"]):
                    continue
                d = delong_test(np.array(ra["y_true"]),
                                 np.array(ra["y_score"]),
                                 np.array(rb["y_score"]))
                delong_results.append({
                    "comparison": f"{gnn} vs MLP ({modal})", **d
                })

    for d in delong_results:
        logger.info("  %s: ΔAUC=%+.4f z=%+.3f p=%.4f",
                    d["comparison"], d["delta"], d["z"], d["p"])

    write_results(task, {
        "params": vars(args),
        "summary": summary,
        "delong": delong_results,
        "n_results": len(results),
    })
    logger.info("Writing: %s/%s.json", RESULTS_DIR, task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
