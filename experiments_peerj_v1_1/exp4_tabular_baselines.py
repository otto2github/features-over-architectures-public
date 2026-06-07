#!/usr/bin/env python
"""
exp4_tabular_baselines.py  -  Editor-mock #5 (stronger tabular baselines) + #4 (TOST data)
==========================================================================================
回应更严格 mock 的两点:
  (#5) 表格基线未加类权重,而神经/GNN 用 pos_weight≈30 -> apples-to-oranges。
       本脚本补强表格基线(全部 class-weighted),在 M5/M10/M11 × 5 seed、验证集选参:
         - class-weighted Logistic Regression (L2)
         - XGBoost (scale_pos_weight + 验证集小网格 max_depth/lr/n_estimators)
         - LightGBM (class-weighted + 同款小网格)
       报告 5 指标 (AUC/AP/F1@val-best/P@5%/R@10FPR),与正文 Table 3 列一致。
  (#4) 同一次 dump 5 个 GNN/MLP 在 M11 的 per-seed test AUC (co-run consistency set),
       供 Claude 计算 TOST 等价检验 (MLP vs SAGE / MLP vs GAT)。

运行 (4090):
  cd ~/cyq/thesis_project/scripts/p7_gnn
  THESIS_PROJECT_ROOT=~/cyq/thesis_project \
  nohup ~/cyq/thesis_env/.venv/bin/python \
      ~/cyq/peerj_experiments/exp4_tabular_baselines.py \
      --label-col fraud_v08_strict \
      > ~/cyq/peerj_experiments/exp4_tabular.log 2>&1 &

可用 --skip-gnn 只跑表格 (快, ~15-25min CPU);--skip-tabular 只跑 GNN co-run。
预计: 表格 ~15-25min; GNN co-run ~30-40min。

依赖: 若缺 xgboost / lightgbm:
  cd ~/cyq/thesis_env && uv add xgboost lightgbm
  (或 ~/cyq/thesis_env/.venv/bin/pip install xgboost lightgbm)
缺失会自动跳过并在日志标注。
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, roc_curve,
)

SEEDS      = [42, 123, 456, 789, 1024]
MODALS     = ["M5", "M10", "M11"]
GNN_MODELS = ["MLP", "GCN", "GAT", "SAGE", "RGCN"]   # baseline 内是 SAGE


# ── 特征列 (复用 baseline 的模态定义) ──────────────────────────────────────────
def get_feature_cols(df, modal):
    prefixes = tuple(base.MODAL_DEFINITIONS[modal])
    return [c for c in df.columns if c.startswith(prefixes)]


# ── 5 指标 (与 Table 3 列一致) ────────────────────────────────────────────────
def compute_metrics(y_true, y_score, y_val_true=None, y_val_score=None):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    # F1@val-best: 在验证集上选 F1 最优阈值, 用到测试集
    if y_val_true is not None and len(np.unique(y_val_true)) > 1:
        y_val_true = np.asarray(y_val_true); y_val_score = np.asarray(y_val_score)
        cand = np.unique(np.quantile(y_val_score, np.linspace(0.50, 0.999, 60)))
        best_f1, best_th = -1.0, 0.5
        for th in cand:
            f1v = f1_score(y_val_true, (y_val_score >= th).astype(int),
                           zero_division=0)
            if f1v > best_f1:
                best_f1, best_th = f1v, th
        f1_test = f1_score(y_true, (y_score >= best_th).astype(int),
                           zero_division=0)
    else:
        f1_test = float("nan")
    # P@5%: 测试集得分最高 5% 的精确率
    k = max(1, int(round(0.05 * len(y_score))))
    topk = np.argsort(-y_score)[:k]
    p_at_5 = float(y_true[topk].mean())
    # R@10%FPR: FPR=0.10 处的召回
    fpr, tpr, _ = roc_curve(y_true, y_score)
    j = int(np.searchsorted(fpr, 0.10, side="right")) - 1
    j = max(0, min(j, len(tpr) - 1))
    r_at_10 = float(tpr[j])
    return {"auc": float(auc), "ap": float(ap), "f1": float(f1_test),
            "p_at_5": p_at_5, "r_at_10fpr": r_at_10}


def build_xy(df, feat_cols, years, label_col, mu, sd):
    sub = df[df["year"].isin(years)]
    X = ((sub[feat_cols].fillna(0) - mu) / sd).values.astype(np.float32)
    y = sub[label_col].fillna(0).astype(int).values
    return X, y


def agg(metric_dicts):
    """对若干 seed 的 metric dict 求 mean/std。"""
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                  "vals": [float(v) for v in vals]}
    return out


# ── (1) 强表格基线 ────────────────────────────────────────────────────────────
def run_tabular(args, features_df, logger):
    try:
        from xgboost import XGBClassifier; xgb_ok = True
    except Exception as e:
        xgb_ok = False; logger.warning("xgboost 不可用 (%s) -> 跳过", e)
    try:
        from lightgbm import LGBMClassifier; lgb_ok = True
    except Exception as e:
        lgb_ok = False; logger.warning("lightgbm 不可用 (%s) -> 跳过", e)

    results = {}
    for modal in MODALS:
        feat_cols = get_feature_cols(features_df, modal)
        tr = features_df[features_df["year"].isin(SPLIT_TRAIN_YEARS)]
        mu = tr[feat_cols].mean(); sd = tr[feat_cols].std().replace(0, 1)
        X_tr, y_tr = build_xy(features_df, feat_cols, SPLIT_TRAIN_YEARS, args.label_col, mu, sd)
        X_va, y_va = build_xy(features_df, feat_cols, SPLIT_VAL_YEARS,   args.label_col, mu, sd)
        X_te, y_te = build_xy(features_df, feat_cols, SPLIT_TEST_YEARS,  args.label_col, mu, sd)
        n_pos = int(y_tr.sum()); n_neg = int((y_tr == 0).sum())
        spw = n_neg / max(1, n_pos)
        logger.info("── modal=%s dim=%d  train_pos=%d spw=%.1f", modal,
                    len(feat_cols), n_pos, spw)

        # --- class-weighted Logistic Regression (确定性) ---
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                 max_iter=2000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        m = compute_metrics(y_te, clf.predict_proba(X_te)[:, 1],
                            y_va, clf.predict_proba(X_va)[:, 1])
        results[f"LogReg_weighted|{modal}"] = agg([m] * len(SEEDS))  # 确定性, std=0
        logger.info("   LogReg_weighted  AUC=%.4f", m["auc"])

        # --- XGBoost (scale_pos_weight + 验证集小网格, 选参后 5 seed) ---
        if xgb_ok:
            grid = [(d, lr, n) for d in (4, 6) for lr in (0.05, 0.1)
                    for n in (400, 800)]
            best_cfg, best_vauc = None, -1.0
            for (d, lr, n) in grid:
                xc = XGBClassifier(max_depth=d, learning_rate=lr, n_estimators=n,
                                   subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=2, scale_pos_weight=spw,
                                   eval_metric="auc", n_jobs=8, verbosity=0,
                                   random_state=42, tree_method="hist")
                xc.fit(X_tr, y_tr)
                vauc = roc_auc_score(y_va, xc.predict_proba(X_va)[:, 1])
                if vauc > best_vauc:
                    best_vauc, best_cfg = vauc, (d, lr, n)
            logger.info("   XGB best cfg (val AUC=%.4f): depth=%d lr=%.2f n=%d",
                        best_vauc, *best_cfg)
            md = []
            for seed in SEEDS:
                d, lr, n = best_cfg
                xc = XGBClassifier(max_depth=d, learning_rate=lr, n_estimators=n,
                                   subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=2, scale_pos_weight=spw,
                                   eval_metric="auc", n_jobs=8, verbosity=0,
                                   random_state=seed, tree_method="hist")
                xc.fit(X_tr, y_tr)
                md.append(compute_metrics(y_te, xc.predict_proba(X_te)[:, 1],
                                          y_va, xc.predict_proba(X_va)[:, 1]))
            results[f"XGBoost_weighted|{modal}"] = agg(md)
            results[f"XGBoost_weighted|{modal}"]["_cfg"] = {
                "max_depth": best_cfg[0], "learning_rate": best_cfg[1],
                "n_estimators": best_cfg[2], "val_auc": float(best_vauc)}
            logger.info("   XGBoost_weighted AUC=%.4f±%.4f",
                        results[f"XGBoost_weighted|{modal}"]["auc"]["mean"],
                        results[f"XGBoost_weighted|{modal}"]["auc"]["std"])

        # --- LightGBM (class_weight balanced + 同款小网格) ---
        if lgb_ok:
            grid = [(d, lr, n) for d in (-1, 6) for lr in (0.05, 0.1)
                    for n in (400, 800)]
            best_cfg, best_vauc = None, -1.0
            for (d, lr, n) in grid:
                lc = LGBMClassifier(max_depth=d, learning_rate=lr, n_estimators=n,
                                    subsample=0.8, colsample_bytree=0.8,
                                    min_child_samples=20, class_weight="balanced",
                                    n_jobs=8, random_state=42, verbosity=-1)
                lc.fit(X_tr, y_tr)
                vauc = roc_auc_score(y_va, lc.predict_proba(X_va)[:, 1])
                if vauc > best_vauc:
                    best_vauc, best_cfg = vauc, (d, lr, n)
            logger.info("   LGBM best cfg (val AUC=%.4f): depth=%d lr=%.2f n=%d",
                        best_vauc, *best_cfg)
            md = []
            for seed in SEEDS:
                d, lr, n = best_cfg
                lc = LGBMClassifier(max_depth=d, learning_rate=lr, n_estimators=n,
                                    subsample=0.8, colsample_bytree=0.8,
                                    min_child_samples=20, class_weight="balanced",
                                    n_jobs=8, random_state=seed, verbosity=-1)
                lc.fit(X_tr, y_tr)
                md.append(compute_metrics(y_te, lc.predict_proba(X_te)[:, 1],
                                          y_va, lc.predict_proba(X_va)[:, 1]))
            results[f"LightGBM_weighted|{modal}"] = agg(md)
            results[f"LightGBM_weighted|{modal}"]["_cfg"] = {
                "max_depth": best_cfg[0], "learning_rate": best_cfg[1],
                "n_estimators": best_cfg[2], "val_auc": float(best_vauc)}
            logger.info("   LightGBM_weighted AUC=%.4f±%.4f",
                        results[f"LightGBM_weighted|{modal}"]["auc"]["mean"],
                        results[f"LightGBM_weighted|{modal}"]["auc"]["std"])

    return results


# ── (2) GNN co-run: per-seed M11 AUC (供 TOST) ────────────────────────────────
def run_gnn_m11(args, features_df, node_idx_df, edges_df, device, logger):
    logger.info("=" * 64)
    logger.info("(2) GNN co-run @ M11 (per-seed AUC for TOST + consistency)")
    logger.info("=" * 64)
    per_seed = {m: [] for m in GNN_MODELS}
    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("── [seed=%d] build M11 ──", seed)
        yearly = prepare_yearly_data(features_df, node_idx_df, edges_df,
                                     "M11", args.label_col, logger)
        for model_name in GNN_MODELS:
            try:
                r = train_and_eval(model_name, "M11", yearly,
                                   SPLIT_TRAIN_YEARS, SPLIT_VAL_YEARS,
                                   SPLIT_TEST_YEARS, args, logger, device)
                auc = float(r["metrics"]["auc"])
            except Exception as e:
                logger.error("   ERROR %s/M11/seed=%d: %s", model_name, seed, e)
                auc = float("nan")
            per_seed[model_name].append(auc)
            logger.info("   %s/M11/seed=%d AUC=%.4f", model_name, seed, auc)
    summary = {}
    for m in GNN_MODELS:
        arr = per_seed[m]
        summary[m] = {"aucs": arr, "mean": float(np.nanmean(arr)),
                      "std": float(np.nanstd(arr))}
        logger.info("  >> %s M11 co-run: %.4f ± %.4f", m,
                    summary[m]["mean"], summary[m]["std"])
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label-col", default="fraud_v08_strict")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-tabular", action="store_true")
    p.add_argument("--skip-gnn", action="store_true")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    logger = setup_logger("tabular_baselines")
    t0 = time.time()
    logger.info("EXP4 Stronger tabular baselines + GNN co-run for TOST")
    logger.info("Device: %s | Seeds: %s | Modals: %s", device, SEEDS, MODALS)

    features_df = load_node_features_v11()
    out = {"experiment": "tabular_baselines_and_tost", "seeds": SEEDS,
           "label_col": args.label_col, "modals": MODALS}

    if not args.skip_tabular:
        out["tabular"] = run_tabular(args, features_df, logger)

    if not args.skip_gnn:
        edges_df    = load_edge_index()
        node_idx_df = load_node_index()
        logger.info("features=%s edges=%s nodes=%s",
                    features_df.shape, edges_df.shape, node_idx_df.shape)
        out["gnn_m11_per_seed"] = run_gnn_m11(
            args, features_df, node_idx_df, edges_df, device, logger)

    out["runtime_min"] = round((time.time() - t0) / 60, 1)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"tabular_baselines_{args.label_col}_{ts}.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("\nSaved -> %s", outpath)
    logger.info("Total: %.1f min", (time.time() - t0) / 60)

    # 关键摘要
    logger.info("\n" + "=" * 64)
    logger.info("KEY SUMMARY")
    if "tabular" in out:
        logger.info("-- Tabular (class-weighted) test AUC mean±std --")
        for key, v in out["tabular"].items():
            logger.info("  %-26s AUC=%.4f±%.4f", key,
                        v["auc"]["mean"], v["auc"]["std"])
    if "gnn_m11_per_seed" in out:
        logger.info("-- GNN co-run @ M11 (for TOST) --")
        for m, v in out["gnn_m11_per_seed"].items():
            logger.info("  %-6s %.4f ± %.4f  seeds=%s", m, v["mean"], v["std"],
                        [f"{a:.4f}" for a in v["aucs"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
