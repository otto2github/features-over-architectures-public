#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1.0 primary tabular baselines (CPU only)

Four fixed, explicit standard tabular learners:
- Lasso = L1-penalized logistic regression
- Ridge = L2-penalized logistic regression
- RandomForest
- XGBoost

Across M5/M10/M11 and seeds {42,123,456,789,1024}: 60 runs.

Protocol:
- same frozen primary label/risk set as neural/GNN benchmark;
- train 2010–2018, validation 2019–2020, test 2021–2022;
- no class weighting in this MAIN tabular baseline suite;
- L1/L2 logistic inputs use train-only standardization;
- tree inputs use raw numeric features with missing -> 0;
- F1 threshold selected on validation only, frozen on test;
- test predictions persisted for later paired/bootstrap analyses;
- explicit model hyperparameters (no library-default ambiguity).

No GPU, no neural training, no modification of existing experiment artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEEDS = [42, 123, 456, 789, 1024]
MODELS = ["Lasso", "Ridge", "RandomForest", "XGBoost"]
MODALS = ["M5", "M10", "M11"]

def build_model(name, seed, n_jobs):
    if name == "Lasso":
        return LogisticRegression(
            solver="liblinear",
            C=1.0,
            l1_ratio=1.0,
            class_weight=None,
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
        ), {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "l1_ratio": 1.0,
            "solver": "liblinear",
            "C": 1.0,
            "class_weight": None,
            "max_iter": 5000,
            "tol": 1e-4,
        }

    if name == "Ridge":
        return LogisticRegression(
            solver="liblinear",
            C=1.0,
            l1_ratio=0.0,
            class_weight=None,
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
        ), {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "l1_ratio": 0.0,
            "solver": "liblinear",
            "C": 1.0,
            "class_weight": None,
            "max_iter": 5000,
            "tol": 1e-4,
        }

    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            n_jobs=n_jobs,
            random_state=seed,
        ), {
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 500,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
        }

    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError(
                "XGBoost is not installed in the PeerJ venv. "
                "Install xgboost in that venv before running this script."
            ) from e

        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        ), {
            "estimator": "xgboost.XGBClassifier",
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "class_weight": None,
        }

    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "cyq/peerj_141707_rerun_v1",
    )
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    root = args.root
    sys.path.insert(0, str(root / "code"))
    import formal_rerun_core as frc

    data_dir = root / "data"
    out_root = root / "results/tabular_primary_v1"
    out_root.mkdir(parents=True, exist_ok=True)

    df = frc.load_joined(data_dir, "label_v1_strict_ab_primary")

    eligible = df["target"].notna().to_numpy()
    years = df["year"].to_numpy()
    yall = df["target"].fillna(0).astype(int).to_numpy()

    tr = np.where(eligible & np.isin(years, frc.TRAIN_YEARS))[0]
    va = np.where(eligible & np.isin(years, frc.VAL_YEARS))[0]
    te = np.where(eligible & np.isin(years, frc.TEST_YEARS))[0]

    assert (len(tr), int(yall[tr].sum())) == (24106, 254)
    assert (len(va), int(yall[va].sum())) == (7309, 40)
    assert (len(te), int(yall[te].sum())) == (8435, 20)

    per_modal = {}

    for modal in MODALS:
        cols = frc.select_cols(df, modal)
        expected = {"M5": 104, "M10": 122, "M11": 129}[modal]
        assert len(cols) == expected, (modal, len(cols), expected)

        mu, sd = frc.fit_scaler(df, cols)
        x_scaled = frc.transformed(df, cols, mu, sd)
        x_raw = df[cols].fillna(0).to_numpy(np.float32)

        per_modal[modal] = {
            "cols": cols,
            "scaled": x_scaled,
            "raw": x_raw,
        }

    print("PeerJ 141707 | primary tabular baselines")
    print("GPU used = false")
    print("train =", len(tr), "positive =", int(yall[tr].sum()))
    print("validation =", len(va), "positive =", int(yall[va].sum()))
    print("test =", len(te), "positive =", int(yall[te].sum()))
    print("dimensions =", {m: len(per_modal[m]["cols"]) for m in MODALS})

    for seed in SEEDS:
        for modal in MODALS:
            for model_name in MODELS:
                run_id = f"tabular_{model_name}_{modal}_seed{seed}"
                out = out_root / run_id
                result_path = out / "result.json"

                if result_path.is_file():
                    print("SKIP completed", run_id, flush=True)
                    continue

                out.mkdir(parents=True, exist_ok=True)

                model, config = build_model(
                    model_name,
                    seed,
                    args.n_jobs,
                )

                if model_name in {"Lasso", "Ridge"}:
                    X = per_modal[modal]["scaled"]
                    preprocessing = (
                        "missing->0; mean/std fit on eligible 2010-2018 "
                        "training rows only"
                    )
                else:
                    X = per_modal[modal]["raw"]
                    preprocessing = "missing->0; no scaling for tree learner"

                started = time.time()

                model.fit(X[tr], yall[tr])

                val_score = model.predict_proba(X[va])[:, 1]
                val_auc = float(roc_auc_score(yall[va], val_score))

                threshold = frc.f1_val_threshold(
                    yall[va],
                    val_score,
                )

                test_score = model.predict_proba(X[te])[:, 1]
                met = frc.metrics(
                    yall[te],
                    test_score,
                    threshold,
                )

                pred = pd.DataFrame({
                    "company_code": df.iloc[te]["company_code"].to_numpy(),
                    "fiscal_year": df.iloc[te]["year"].to_numpy(),
                    "y_true": yall[te],
                    "score": test_score,
                })
                pred.to_parquet(
                    out / "predictions.parquet",
                    index=False,
                )

                result = {
                    "run_id": run_id,
                    "model": model_name,
                    "modal": modal,
                    "seed": seed,
                    "label_col": "label_v1_strict_ab_primary",
                    "feature_dim": len(per_modal[modal]["cols"]),
                    "class_weight_mode": "none",
                    "model_config": config,
                    "preprocessing": preprocessing,
                    "train_n": len(tr),
                    "train_pos": int(yall[tr].sum()),
                    "train_neg": int(len(tr) - yall[tr].sum()),
                    "validation_n": len(va),
                    "validation_pos": int(yall[va].sum()),
                    "test_n": len(te),
                    "test_pos": int(yall[te].sum()),
                    "val_auc": val_auc,
                    "val_threshold": float(threshold),
                    "metrics": met,
                    "test_evaluated": True,
                    "runtime_seconds": time.time() - started,
                }

                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                print(
                    "DONE",
                    run_id,
                    "val_auc=",
                    round(val_auc, 6),
                    "test_auc=",
                    round(met["roc_auc"], 6),
                    flush=True,
                )

    # Aggregate.
    rows = []
    files = sorted(out_root.glob("*/result.json"))
    assert len(files) == 60, len(files)

    for p in files:
        j = json.loads(p.read_text())
        m = j["metrics"]
        rows.append({
            "run_id": j["run_id"],
            "model": j["model"],
            "modal": j["modal"],
            "seed": j["seed"],
            "feature_dim": j["feature_dim"],
            "val_auc": j["val_auc"],
            "val_threshold": j["val_threshold"],
            "roc_auc": m["roc_auc"],
            "ap": m["ap"],
            "f1_val_threshold": m["f1_val_threshold"],
            "p_at_5pct": m["p_at_5pct"],
            "r_at_10fpr": m["r_at_10fpr"],
            "runtime_seconds": j["runtime_seconds"],
        })

    per_seed = pd.DataFrame(rows).sort_values(
        ["modal", "model", "seed"]
    )
    per_seed.to_csv(
        root / "results/tabular_primary_v1_summary.per_seed.csv",
        index=False,
    )

    summary = []
    for (model, modal), g in per_seed.groupby(["model", "modal"]):
        z = {
            "model": model,
            "modal": modal,
            "n_seeds": len(g),
        }
        for c in [
            "roc_auc",
            "ap",
            "f1_val_threshold",
            "p_at_5pct",
            "r_at_10fpr",
        ]:
            z[c + "_mean"] = float(g[c].mean())
            z[c + "_sd"] = float(g[c].std(ddof=1))
        summary.append(z)

    pd.DataFrame(summary).sort_values(
        ["modal", "roc_auc_mean"],
        ascending=[True, False],
    ).to_csv(
        root / "results/tabular_primary_v1_summary.csv",
        index=False,
    )

    print("result_json =", len(files))
    print("STATUS: TABULAR_PRIMARY_60_RUNS_COMPLETE")

if __name__ == "__main__":
    main()
