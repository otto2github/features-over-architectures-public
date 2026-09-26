#!/usr/bin/env python3
"""
PeerJ CS 141707 — final calibration rerun for MLP × M11 under Label v1.0.

Purpose
-------
Re-train ONLY the already-defined primary MLP × M11 condition across the same
five seeds, using the frozen corrected protocol, because the original formal
rerun persisted test scores but not validation scores required for leakage-free
post-hoc calibration.

For each seed:
- identical MLP architecture / optimizer / class-weighted BCE / early stopping;
- same train 2010–2018, validation 2019–2020, mature test 2021–2022;
- persist validation and test raw sigmoid scores;
- fit a two-parameter Platt recalibration model on VALIDATION scores only;
- evaluate raw and Platt-calibrated probabilities on TEST only;
- verify reproduced test ROC-AUC against the existing core_v1 MLP×M11 result.

Final outputs:
- per-seed calibration metrics;
- five-seed mean ± SD;
- 2,000-replicate company-cluster bootstrap CI for mean calibration metrics;
- 15-bin reliability-curve data for the five-seed mean probabilities;
- no calibration fit uses test labels.

This is intended as the LAST GPU-dependent experiment unless a later manuscript
audit identifies a specific unresolved reviewer requirement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

SEEDS = [42, 123, 456, 789, 1024]
PRIMARY = "label_v1_strict_ab_primary"
EPS = 1e-7


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def logit(p):
    p = clip_prob(p)
    return np.log(p) - np.log1p(-p)


def fit_platt(y, raw_prob):
    """
    Fit P(y=1 | score) = sigmoid(a * logit(raw_prob) + b)
    on validation data only.
    """
    y = np.asarray(y, dtype=np.float64)
    z = logit(raw_prob)

    def objective(theta):
        a, b = theta
        eta = a * z + b
        # Bernoulli negative log likelihood, numerically stable.
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

    res = minimize(
        objective,
        x0=np.array([1.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"PLATT_OPTIMIZATION_FAILED:{res.message}")

    a, b = map(float, res.x)
    return a, b


def apply_platt(raw_prob, a, b):
    return expit(a * logit(raw_prob) + b)


def weighted_mean(x, w):
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    den = float(w.sum())
    return float(np.sum(w * x) / den)


def weighted_calibration_metrics(y, p, w=None, n_bins=15):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)

    if w is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)

    if len(y) != len(p) or len(y) != len(w):
        raise RuntimeError("CALIBRATION_LENGTH_MISMATCH")

    total_w = float(w.sum())
    if total_w <= 0:
        raise RuntimeError("ZERO_TOTAL_WEIGHT")

    brier = weighted_mean((p - y) ** 2, w)
    mean_pred = weighted_mean(p, w)
    base_rate = weighted_mean(y, w)

    # Weighted Bernoulli log loss.
    ll = weighted_mean(
        -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        w,
    )

    # Equal-width ECE/MCE over [0,1].
    # p=1 goes into last bin.
    bin_idx = np.minimum((p * n_bins).astype(int), n_bins - 1)

    ece = 0.0
    mce = 0.0
    nonempty = 0
    for b in range(n_bins):
        m = bin_idx == b
        if not np.any(m):
            continue
        wb = w[m]
        sw = float(wb.sum())
        if sw <= 0:
            continue
        nonempty += 1
        conf = weighted_mean(p[m], wb)
        obs = weighted_mean(y[m], wb)
        gap = abs(conf - obs)
        ece += (sw / total_w) * gap
        mce = max(mce, gap)

    # AUC/AP are included to verify that monotone Platt scaling does not alter
    # ranking unless the fitted slope is negative.
    auc = float(roc_auc_score(y, p, sample_weight=w))
    ap = float(average_precision_score(y, p, sample_weight=w))

    unc = base_rate * (1.0 - base_rate)
    brier_skill = (
        float(1.0 - brier / unc)
        if unc > 0
        else float("nan")
    )

    return {
        "brier": brier,
        "log_loss": ll,
        "ece15": float(ece),
        "mce15": float(mce),
        "mean_predicted_probability": mean_pred,
        "base_rate": base_rate,
        "brier_skill_vs_constant_base_rate": brier_skill,
        "roc_auc": auc,
        "ap": ap,
        "nonempty_bins": int(nonempty),
    }


def reliability_table(y, p_raw, p_platt, n_bins=15):
    """
    Reliability table for five-seed mean probabilities.
    Binning is based separately for each probability vector, but rows are
    returned on common equal-width [0,1] bin edges.
    """
    y = np.asarray(y, dtype=int)
    raw = clip_prob(p_raw)
    platt = clip_prob(p_platt)

    rows = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins

        def summarize(p):
            idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
            m = idx == b
            if not np.any(m):
                return 0, np.nan, np.nan
            return (
                int(m.sum()),
                float(np.mean(p[m])),
                float(np.mean(y[m])),
            )

        nr, mr, oraw = summarize(raw)
        np_, mp, oplatt = summarize(platt)

        rows.append({
            "bin": b + 1,
            "lower": lo,
            "upper": hi,
            "raw_n": nr,
            "raw_mean_pred": mr,
            "raw_observed_rate": oraw,
            "platt_n": np_,
            "platt_mean_pred": mp,
            "platt_observed_rate": oplatt,
        })
    return pd.DataFrame(rows)


def discover_core_reference(root, seed):
    """
    Locate the already-completed canonical core_v1 MLP×M11 result by content,
    without assuming a run-directory naming convention.
    """
    matches = []
    for p in (root / "results/core_v1").glob("*/result.json"):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if (
            j.get("model") == "MLP"
            and j.get("modal") == "M11"
            and int(j.get("seed", -1)) == seed
            and j.get("label_col") == PRIMARY
        ):
            matches.append((p, j))

    if len(matches) != 1:
        raise RuntimeError(
            f"CORE_REFERENCE_MATCH_COUNT:seed={seed}:count={len(matches)}"
        )
    return matches[0]


def train_one_seed(root, df, seed, device, epochs, patience, outdir):
    import torch
    import torch.nn as nn

    code_dir = root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import formal_rerun_core as frc

    frc.set_seed(seed)

    cols = frc.select_cols(df, "M11")
    if len(cols) != 129:
        raise RuntimeError(f"M11_DIM_MISMATCH:{len(cols)}")

    mu, sd = frc.fit_scaler(df, cols)
    X = frc.transformed(df, cols, mu, sd)

    eligible = df["target"].notna().to_numpy()
    years = df["year"].to_numpy()
    yall = df["target"].fillna(0).astype(int).to_numpy()

    tr = np.where(
        eligible & np.isin(years, frc.TRAIN_YEARS)
    )[0]
    va = np.where(
        eligible & np.isin(years, frc.VAL_YEARS)
    )[0]
    te = np.where(
        eligible & np.isin(years, frc.TEST_YEARS)
    )[0]

    assert (len(tr), int(yall[tr].sum())) == (24106, 254)
    assert (len(va), int(yall[va].sum())) == (7309, 40)
    assert (len(te), int(yall[te].sum())) == (8435, 20)

    dev = torch.device(device)
    model = frc.make_model("MLP", len(cols), 64).to(dev)

    posw, neg, pos = frc.dynamic_pos_weight(df)
    if abs(posw - 93.90551181102362) > 1e-10:
        raise RuntimeError(f"POS_WEIGHT_MISMATCH:{posw}")

    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(posw, device=dev)
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    xt = torch.from_numpy(X[tr]).to(dev)
    yt = torch.from_numpy(
        yall[tr].astype(np.float32)
    ).to(dev)
    xv = torch.from_numpy(X[va]).to(dev)

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    stale = 0

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(xt)
        loss = crit(logits, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(xv)).cpu().numpy()

        val_auc = float(roc_auc_score(yall[va], val_prob))
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu()),
            "val_auc": val_auc,
        })

        if val_auc > best_auc + 1e-8:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("NO_BEST_STATE")

    model.load_state_dict(best_state)
    model.to(dev)
    model.eval()

    xtest = torch.from_numpy(X[te]).to(dev)
    with torch.no_grad():
        val_raw = torch.sigmoid(model(xv)).cpu().numpy().astype(np.float64)
        test_raw = torch.sigmoid(model(xtest)).cpu().numpy().astype(np.float64)

    # Validation-only Platt fit.
    a, b = fit_platt(yall[va], val_raw)
    val_platt = apply_platt(val_raw, a, b)
    test_platt = apply_platt(test_raw, a, b)

    raw_metrics = weighted_calibration_metrics(
        yall[te], test_raw, n_bins=15
    )
    platt_metrics = weighted_calibration_metrics(
        yall[te], test_platt, n_bins=15
    )

    # Compare to the already-completed canonical core_v1 result.
    ref_path, ref = discover_core_reference(root, seed)
    ref_auc = float(ref["metrics"]["roc_auc"])
    auc_delta = raw_metrics["roc_auc"] - ref_auc
    if abs(auc_delta) > 1e-6:
        raise RuntimeError(
            f"CORE_REPRODUCTION_AUC_MISMATCH:"
            f"seed={seed}:new={raw_metrics['roc_auc']}:"
            f"old={ref_auc}:delta={auc_delta}"
        )

    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "company_code": df.iloc[va]["company_code"].to_numpy(),
        "fiscal_year": df.iloc[va]["year"].to_numpy(),
        "y_true": yall[va],
        "raw_probability": val_raw,
        "platt_probability": val_platt,
    }).to_parquet(
        outdir / "validation_predictions.parquet",
        index=False,
    )

    pd.DataFrame({
        "company_code": df.iloc[te]["company_code"].to_numpy(),
        "fiscal_year": df.iloc[te]["year"].to_numpy(),
        "y_true": yall[te],
        "raw_probability": test_raw,
        "platt_probability": test_platt,
    }).to_parquet(
        outdir / "test_predictions.parquet",
        index=False,
    )

    result = {
        "run_id": f"calibration_mlp_m11_seed{seed}",
        "model": "MLP",
        "modal": "M11",
        "seed": seed,
        "label_col": PRIMARY,
        "feature_dim": 129,
        "train_n": len(tr),
        "train_pos": int(yall[tr].sum()),
        "validation_n": len(va),
        "validation_pos": int(yall[va].sum()),
        "test_n": len(te),
        "test_pos": int(yall[te].sum()),
        "pos_weight": float(posw),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc),
        "epochs_run": len(history),
        "platt": {
            "fit_partition": "validation_2019_2020_only",
            "predictor": "logit(raw_sigmoid_probability)",
            "a": float(a),
            "b": float(b),
            "validation_raw_auc": float(
                roc_auc_score(yall[va], val_raw)
            ),
            "validation_platt_auc": float(
                roc_auc_score(yall[va], val_platt)
            ),
        },
        "test_raw": raw_metrics,
        "test_platt": platt_metrics,
        "core_reference": {
            "result_json": str(ref_path.relative_to(root)),
            "reference_test_auc": ref_auc,
            "rerun_test_auc": raw_metrics["roc_auc"],
            "absolute_auc_delta": abs(float(auc_delta)),
            "reproduction_tolerance": 1e-6,
            "reproduction_pass": True,
        },
        "protocol": {
            "train_years": list(frc.TRAIN_YEARS),
            "validation_years": list(frc.VAL_YEARS),
            "test_years": list(frc.TEST_YEARS),
            "architecture": "same formal_rerun_core MLP: 129->64->64->1",
            "optimizer": "Adam",
            "lr": 5e-4,
            "max_epochs": epochs,
            "early_stopping": "validation ROC-AUC; patience 10",
            "loss": "BCEWithLogitsLoss with dynamic train neg/pos pos_weight",
            "test_used_for_training_or_calibration": False,
            "calibration": "two-parameter Platt scaling fit on validation only",
            "ece_mce": "15 equal-width probability bins",
        },
        "history": history,
    }

    (outdir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    del model, opt, xt, yt, xv, xtest
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def bootstrap(root, result_root, meta_dir, n_boot):
    runs = []
    for seed in SEEDS:
        od = result_root / f"seed{seed}"
        j = json.loads((od / "result.json").read_text())
        p = pd.read_parquet(
            od / "test_predictions.parquet"
        ).sort_values(
            ["company_code", "fiscal_year"]
        ).reset_index(drop=True)
        runs.append((j, p))

    master = runs[0][1]
    keys = list(zip(
        master["company_code"].astype(str),
        master["fiscal_year"].astype(int),
    ))
    y = master["y_true"].to_numpy(int)

    if (len(y), int(y.sum())) != (8435, 20):
        raise RuntimeError("BOOTSTRAP_TEST_COUNTS_MISMATCH")

    for _, p in runs[1:]:
        k = list(zip(
            p["company_code"].astype(str),
            p["fiscal_year"].astype(int),
        ))
        if k != keys:
            raise RuntimeError("CALIBRATION_KEY_MISMATCH")
        if not np.array_equal(p["y_true"].to_numpy(int), y):
            raise RuntimeError("CALIBRATION_LABEL_MISMATCH")

    raw_scores = [
        p["raw_probability"].to_numpy(float)
        for _, p in runs
    ]
    platt_scores = [
        p["platt_probability"].to_numpy(float)
        for _, p in runs
    ]

    cat = pd.Categorical(master["company_code"].astype(str))
    cluster_idx = cat.codes.astype(np.int32)
    n_clusters = len(cat.categories)

    metric_names = [
        "brier",
        "log_loss",
        "ece15",
        "mce15",
        "mean_predicted_probability",
        "brier_skill_vs_constant_base_rate",
        "roc_auc",
        "ap",
    ]

    def mean_metric(scores, w):
        rows = [
            weighted_calibration_metrics(
                y, s, w=w, n_bins=15
            )
            for s in scores
        ]
        return np.array([
            np.mean([r[n] for r in rows])
            for n in metric_names
        ], dtype=float)

    one = np.ones(len(y), dtype=float)
    obs_raw = mean_metric(raw_scores, one)
    obs_platt = mean_metric(platt_scores, one)

    br = np.empty((n_boot, len(metric_names)), dtype=float)
    bp = np.empty((n_boot, len(metric_names)), dtype=float)

    rng = np.random.default_rng(141707)
    for b in range(n_boot):
        while True:
            sampled = rng.integers(
                0, n_clusters, size=n_clusters
            )
            counts = np.bincount(
                sampled, minlength=n_clusters
            )
            w = counts[cluster_idx].astype(float)
            if np.sum(w*y) > 0 and np.sum(w*(1-y)) > 0:
                break

        br[b] = mean_metric(raw_scores, w)
        bp[b] = mean_metric(platt_scores, w)

        if b == 0 or (b + 1) % 100 == 0:
            print(
                f"calibration bootstrap {b+1}/{n_boot}",
                flush=True,
            )

    rows = []
    for condition, obs, arr in [
        ("raw_sigmoid", obs_raw, br),
        ("platt_validation_only", obs_platt, bp),
    ]:
        row = {
            "condition": condition,
            "n_seeds": 5,
            "n_boot": n_boot,
            "cluster_unit": "company_code",
            "test_n": len(y),
            "test_pos": int(y.sum()),
            "company_clusters": n_clusters,
        }
        for i, name in enumerate(metric_names):
            row[name + "_observed_mean"] = float(obs[i])
            row[name + "_ci_lo"] = float(
                np.quantile(arr[:, i], 0.025)
            )
            row[name + "_ci_hi"] = float(
                np.quantile(arr[:, i], 0.975)
            )
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        meta_dir / "calibration_company_cluster_bootstrap_ci.csv",
        index=False,
    )

    # Five-seed mean score reliability data for a compact figure.
    mean_raw = np.mean(np.vstack(raw_scores), axis=0)
    mean_platt = np.mean(np.vstack(platt_scores), axis=0)
    reliability_table(
        y, mean_raw, mean_platt, n_bins=15
    ).to_csv(
        meta_dir / "reliability_curve_15bin.csv",
        index=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "cyq/peerj_141707_rerun_v1",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    root = args.root.resolve()
    allowed = (Path.home() / "cyq").resolve()
    if allowed != root and allowed not in root.parents:
        raise RuntimeError("REFUSE_WRITE_OUTSIDE_HOME_CYQ")

    code_dir = root / "code"
    sys.path.insert(0, str(code_dir))
    import formal_rerun_core as frc

    data_dir = root / "data"
    result_root = root / "results/calibration_mlp_m11_v1"
    meta_dir = root / "meta/calibration_mlp_m11_v1"
    result_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    df = frc.load_joined(data_dir, PRIMARY)

    print(
        "PeerJ 141707 | final calibration rerun | MLP x M11",
        flush=True,
    )
    print(
        "validation-only Platt calibration; mature test 2021-2022",
        flush=True,
    )

    results = []
    for seed in SEEDS:
        od = result_root / f"seed{seed}"
        rp = od / "result.json"

        if rp.is_file():
            print(f"SKIP completed seed={seed}", flush=True)
            results.append(json.loads(rp.read_text()))
            continue

        print(f"RUN seed={seed}", flush=True)
        started = time.time()
        r = train_one_seed(
            root=root,
            df=df,
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
            outdir=od,
        )
        results.append(r)
        print(
            f"DONE seed={seed} "
            f"best_epoch={r['best_epoch']} "
            f"raw_auc={r['test_raw']['roc_auc']:.6f} "
            f"raw_brier={r['test_raw']['brier']:.6f} "
            f"platt_brier={r['test_platt']['brier']:.6f} "
            f"platt_ece15={r['test_platt']['ece15']:.6f} "
            f"seconds={time.time()-started:.1f}",
            flush=True,
        )

    if len(results) != 5:
        raise RuntimeError(f"CALIBRATION_RESULT_COUNT:{len(results)}")

    # Per-seed table.
    rows = []
    for r in sorted(results, key=lambda x: x["seed"]):
        row = {
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "best_val_auc": r["best_val_auc"],
            "platt_a": r["platt"]["a"],
            "platt_b": r["platt"]["b"],
            "core_auc_delta":
                r["core_reference"]["absolute_auc_delta"],
        }
        for prefix in ["raw", "platt"]:
            m = r[f"test_{prefix}"]
            for c in [
                "brier",
                "log_loss",
                "ece15",
                "mce15",
                "mean_predicted_probability",
                "base_rate",
                "brier_skill_vs_constant_base_rate",
                "roc_auc",
                "ap",
            ]:
                row[f"{prefix}_{c}"] = m[c]
        rows.append(row)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(
        result_root / "calibration_summary.per_seed.csv",
        index=False,
    )

    # Mean ± SD across seeds.
    summary = {
        "model": "MLP",
        "modal": "M11",
        "label": PRIMARY,
        "n_seeds": 5,
        "validation_calibrator": "Platt",
        "test_n": 8435,
        "test_pos": 20,
        "test_base_rate": 20 / 8435,
    }
    for prefix in ["raw", "platt"]:
        for c in [
            "brier",
            "log_loss",
            "ece15",
            "mce15",
            "mean_predicted_probability",
            "brier_skill_vs_constant_base_rate",
            "roc_auc",
            "ap",
        ]:
            col = f"{prefix}_{c}"
            summary[col + "_mean"] = float(per_seed[col].mean())
            summary[col + "_sd"] = float(per_seed[col].std(ddof=1))

    pd.DataFrame([summary]).to_csv(
        result_root / "calibration_summary.csv",
        index=False,
    )

    print("RUN company-cluster bootstrap...", flush=True)
    bootstrap(
        root=root,
        result_root=result_root,
        meta_dir=meta_dir,
        n_boot=args.n_boot,
    )

    report = [
        "PeerJ 141707 | Calibration MLP x M11 v1",
        "label=label_v1_strict_ab_primary",
        "train=2010-2018",
        "validation=2019-2020",
        "test=2021-2022",
        "test_n=8435",
        "test_pos=20",
        "seeds=42,123,456,789,1024",
        "raw_model=class-weighted MLP M11 formal protocol",
        "calibrator=two-parameter Platt scaling",
        "calibrator_fit=validation scores/labels only",
        "test_labels_used_for_calibration=false",
        "ece_mce_bins=15 equal-width",
        f"bootstrap_n={args.n_boot}",
        "bootstrap_cluster=company_code",
        "core_auc_reproduction_required=true",
        "STATUS: CALIBRATION_MLP_M11_V1_COMPLETE",
    ]
    (meta_dir / "calibration_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
