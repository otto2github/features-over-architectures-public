#!/usr/bin/env python
"""
exp2_puregraph_structural.py  -  MAJOR-2 (Reviewer 1)
======================================================
回应审稿意见 MAJOR-2: pure-graph 诊断用 one-hot 退化输入 + 单 seed,
缺结构基线, 无法支撑"弱拓扑信号"结论。

本脚本提供三块证据:
  (A) 多 seed pure-graph: 4 GNN (GCN/GAT/SAGE/RGCN) 在 G6 模态(无特征,
      one-hot identity) 下跑 5 seeds, 报告 mean +/- std (而非单 seed)。
  (B) Node2Vec -> Logistic Regression 结构嵌入基线 (单独的 non-GNN 结构方法)。
  (C) Label Propagation 结构基线 (纯标签在图上传播)。

若 (A)(B)(C) 都接近随机, "拓扑信号弱"的结论从"单一退化诊断"
升级为"多方法一致证据", 显著增强论文。

运行 (4090, 建议 pure-graph 跑完 exp1 之后再跑, 避免抢 GPU):
  cd ~/cyq/thesis_project/scripts/p7_gnn
  THESIS_PROJECT_ROOT=~/cyq/thesis_project \
  ~/cyq/thesis_env/.venv/bin/python \
      ~/cyq/peerj_experiments/exp2_puregraph_structural.py \
      --label-col fraud_v08_strict

预计: pure-graph 多seed ~20-30 min; Node2Vec ~20-40 min; LabelProp ~5 min。
可用 --skip-node2vec / --skip-gnn / --skip-labelprop 分段跑。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEEDS = [42, 123, 456, 789, 1024]
GNN_MODELS = ["GCN", "GAT", "SAGE", "RGCN"]


def firm_id_to_tscode(code):
    """firm_id (6位) -> ts_code (带交易所后缀). 0/3->SZ, 6->SH, 4/8->BJ."""
    code = str(code).zfill(6)
    if code[0] in ("0", "3"):
        return code + ".SZ"
    if code[0] == "6":
        return code + ".SH"
    if code[0] in ("4", "8"):
        return code + ".BJ"
    return code + ".SZ"


# =============================================================================
# (A) 多 seed pure-graph (G6 = 无特征 one-hot identity)
# =============================================================================
def run_pure_graph_multiseed(args, features_df, node_idx_df, edges_df,
                              device, logger):
    logger.info("=" * 64)
    logger.info("(A) Pure-graph diagnostic, MULTI-SEED (G6 = no features)")
    logger.info("=" * 64)
    results = {m: [] for m in GNN_MODELS}
    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("-- [seed=%d] build G6 (pure-graph) --", seed)
        yearly = prepare_yearly_data(
            features_df, node_idx_df, edges_df,
            "G6", args.label_col, logger,
        )
        for model_name in GNN_MODELS:
            try:
                r = train_and_eval(
                    model_name, "G6", yearly,
                    SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS, SPLIT_TEST_YEARS,
                    args, logger, device,
                )
                auc = float(r["metrics"]["auc"])
            except Exception as e:
                logger.error("   ERROR %s/G6/seed=%d: %s", model_name, seed, e)
                auc = float("nan")
            results[model_name].append(auc)
            logger.info("   %s/G6/seed=%d AUC=%.4f", model_name, seed, auc)
    summary = {}
    for m in GNN_MODELS:
        arr = results[m]
        summary[m] = {"aucs": arr,
                      "mean": float(np.nanmean(arr)),
                      "std":  float(np.nanstd(arr))}
        logger.info("  >> %s pure-graph: %.4f +/- %.4f",
                    m, summary[m]["mean"], summary[m]["std"])
    return summary


# =============================================================================
# (B) Node2Vec -> Logistic Regression
# =============================================================================
def run_node2vec_lr(args, features_df, node_idx_df, edges_df, logger):
    logger.info("=" * 64)
    logger.info("(B) Node2Vec -> Logistic Regression structural baseline")
    logger.info("=" * 64)
    from node2vec import Node2Vec
    import networkx as nx

    # 构建无向图 (全图, 用于学结构嵌入)
    logger.info("Building NetworkX graph (%d edges)...", len(edges_df))
    G = nx.Graph()
    src = edges_df["src_idx"].values
    dst = edges_df["dst_idx"].values
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(),
                G.number_of_edges())

    # Node2Vec (维度 64, 适中参数控制时间)
    logger.info("Running Node2Vec (dim=64, walks=10, length=40)...")
    n2v = Node2Vec(G, dimensions=64, walk_length=40, num_walks=10,
                   workers=8, seed=42, quiet=True)
    model = n2v.fit(window=5, min_count=1, batch_words=4)
    logger.info("Node2Vec done.")

    # 嵌入字典: node_idx -> vec
    emb = {}
    for node in G.nodes():
        try:
            emb[node] = model.wv[str(node)]
        except KeyError:
            emb[node] = np.zeros(64, dtype=np.float32)

    # 取 company 节点 firm-year, 用嵌入做特征
    # node_idx_df: company 节点的 node_idx; features_df 有 firm_id/year/label
    # 需要把 firm-year 映射到 node_idx (company)
    company_idx = dict(zip(
        node_idx_df.loc[node_idx_df["node_type"] == "company", "ts_code"],
        node_idx_df.loc[node_idx_df["node_type"] == "company", "node_idx"],
    ))

    id_col = "firm_id" if "firm_id" in features_df.columns else "ts_code"

    def build_xy(year_list):
        X, y = [], []
        sub = features_df[features_df["year"].isin(year_list)]
        for _, row in sub.iterrows():
            nid = company_idx.get(firm_id_to_tscode(row[id_col]))
            if nid is None or nid not in emb:
                continue
            X.append(emb[nid])
            y.append(int(row[args.label_col]))
        return np.array(X), np.array(y)

    seeds_auc = []
    for seed in SEEDS:
        X_tr, y_tr = build_xy(SPLIT_TRAIN_YEARS)
        X_te, y_te = build_xy(SPLIT_TEST_YEARS)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.warning("   degenerate labels, skip seed=%d", seed)
            seeds_auc.append(float("nan")); continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=seed)
        clf.fit(X_tr, y_tr)
        score = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, score))
        seeds_auc.append(auc)
        logger.info("   Node2Vec+LR seed=%d AUC=%.4f", seed, auc)

    out = {"aucs": seeds_auc,
           "mean": float(np.nanmean(seeds_auc)),
           "std":  float(np.nanstd(seeds_auc)),
           "n_train": int(len(y_tr)), "n_test": int(len(y_te))}
    logger.info("  >> Node2Vec+LR: %.4f +/- %.4f", out["mean"], out["std"])
    return out


# =============================================================================
# (C) Label Propagation
# =============================================================================
def run_label_propagation(args, features_df, node_idx_df, edges_df, logger):
    logger.info("=" * 64)
    logger.info("(C) Label Propagation structural baseline")
    logger.info("=" * 64)
    from torch_geometric.nn import LabelPropagation

    n_total = len(node_idx_df)
    company_idx = dict(zip(
        node_idx_df.loc[node_idx_df["node_type"] == "company", "ts_code"],
        node_idx_df.loc[node_idx_df["node_type"] == "company", "node_idx"],
    ))
    id_col = "firm_id" if "firm_id" in features_df.columns else "ts_code"

    # 全图 edge_index (无向)
    src = torch.tensor(edges_df["src_idx"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["dst_idx"].values, dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]),
                              torch.cat([dst, src])])  # 双向

    # train firm-year 的标签作为已知, test 的预测
    y = torch.zeros(n_total, dtype=torch.float)
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    test_idx, test_y = [], []

    tr_sub = features_df[features_df["year"].isin(SPLIT_TRAIN_YEARS)]
    for _, row in tr_sub.iterrows():
        nid = company_idx.get(firm_id_to_tscode(row[id_col]))
        if nid is not None:
            y[nid] = float(row[args.label_col]); train_mask[nid] = True
    te_sub = features_df[features_df["year"].isin(SPLIT_TEST_YEARS)]
    for _, row in te_sub.iterrows():
        nid = company_idx.get(firm_id_to_tscode(row[id_col]))
        if nid is not None:
            test_idx.append(nid); test_y.append(int(row[args.label_col]))

    lp = LabelPropagation(num_layers=3, alpha=0.9)
    yhat = lp(y.unsqueeze(1), edge_index, mask=train_mask)
    scores = yhat[test_idx, 0].numpy()
    test_y = np.array(test_y)
    if len(np.unique(test_y)) < 2:
        logger.warning("   degenerate test labels")
        auc = float("nan")
    else:
        auc = float(roc_auc_score(test_y, scores))
    logger.info("  >> Label Propagation: AUC=%.4f (n_test=%d)", auc, len(test_y))
    return {"auc": auc, "n_test": int(len(test_y))}


# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label-col", default="fraud_v08_strict")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--skip-node2vec", action="store_true")
    p.add_argument("--skip-labelprop", action="store_true")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    logger = setup_logger("puregraph_structural")
    t0 = time.time()
    logger.info("EXP2 Pure-graph + Structural baselines (MAJOR-2)")
    logger.info("Device: %s | Seeds: %s", device, SEEDS)

    features_df = load_node_features_v11()
    edges_df    = load_edge_index()
    node_idx_df = load_node_index()
    logger.info("features=%s edges=%s nodes=%s",
                features_df.shape, edges_df.shape, node_idx_df.shape)

    out = {"experiment": "puregraph_structural_MAJOR2",
           "seeds": SEEDS, "label_col": args.label_col}

    if not args.skip_gnn:
        out["pure_graph_multiseed"] = run_pure_graph_multiseed(
            args, features_df, node_idx_df, edges_df, device, logger)
    if not args.skip_node2vec:
        try:
            out["node2vec_lr"] = run_node2vec_lr(
                args, features_df, node_idx_df, edges_df, logger)
        except Exception as e:
            logger.error("Node2Vec failed: %s", e)
            out["node2vec_lr"] = {"error": str(e)}
    if not args.skip_labelprop:
        try:
            out["label_propagation"] = run_label_propagation(
                args, features_df, node_idx_df, edges_df, logger)
        except Exception as e:
            logger.error("LabelProp failed: %s", e)
            out["label_propagation"] = {"error": str(e)}

    out["runtime_min"] = round((time.time() - t0) / 60, 1)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"puregraph_structural_{args.label_col}_{ts}.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("\nSaved -> %s", outpath)
    logger.info("Total: %.1f min", (time.time() - t0) / 60)

    # 打印关键摘要
    logger.info("\n" + "=" * 64)
    logger.info("KEY SUMMARY (all should be near-random if topology is weak)")
    if "pure_graph_multiseed" in out:
        for m, v in out["pure_graph_multiseed"].items():
            logger.info("  pure-graph %-6s: %.4f +/- %.4f", m, v["mean"], v["std"])
    if "node2vec_lr" in out and "mean" in out["node2vec_lr"]:
        logger.info("  Node2Vec+LR    : %.4f +/- %.4f",
                    out["node2vec_lr"]["mean"], out["node2vec_lr"]["std"])
    if "label_propagation" in out and "auc" in out["label_propagation"]:
        logger.info("  LabelProp      : %.4f", out["label_propagation"]["auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
