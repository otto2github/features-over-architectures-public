#!/usr/bin/env python3
"""
PeerJ 141707 — comprehensive post-hoc company-cluster bootstrap v3 (CPU only).

Consumes 290 persisted test-prediction runs:
  75 primary neural/GNN core
  20 pure-graph
  50 label sensitivity
  25 audit ablation
  60 unweighted tabular
  60 class-weighted tabular

Bootstrap:
- company_code is the resampling cluster;
- all 2021–2022 rows of a company travel together;
- same cluster draw is used across all models/seeds within each label universe;
- 2,000 replicates by default;
- no GPU and no model training.

Primary matched-weight cross-family comparisons use:
  MLP / GCN / GAT / SAGE / RGCN (class-weighted BCE)
  versus
  TABW_Lasso / TABW_Ridge / TABW_RandomForest / TABW_XGBoost
  where the tabular suite uses the same training-set class imbalance ratio.

Unweighted tabular runs are retained as robustness analyses.

Outputs contain aggregate statistics only; no raw firm IDs or scores.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

SEEDS = [42, 123, 456, 789, 1024]
PRIMARY = "label_v1_strict_ab_primary"
AONLY = "label_v1_strict_aonly_sensitivity"
LOOSE = "label_v1_loose_ab_sensitivity"
METRIC_NAMES = ["roc_auc", "ap", "f1", "p_at_5pct", "r_at_10fpr"]

G = {}


def exact_signflip_p(diffs):
    d = np.asarray(diffs, dtype=float)
    obs = abs(float(d.mean()))
    vals = []
    for signs in itertools.product([-1, 1], repeat=len(d)):
        vals.append(abs(float(np.mean(d * np.asarray(signs)))))
    return sum(v >= obs - 1e-15 for v in vals) / len(vals)


def weighted_p_at_budget(y, score, w, budget=.05):
    total = float(w.sum())
    k = max(1.0, math.ceil(total * budget))
    order = np.argsort(-score, kind="mergesort")
    yy = y[order]
    ww = w[order].astype(float)
    cs = np.cumsum(ww)
    before = np.concatenate(([0.0], cs[:-1]))
    take = np.clip(k - before, 0.0, ww)
    denom = float(take.sum())
    return float(np.sum(take * yy) / denom) if denom > 0 else np.nan


def weighted_r_at_fpr(y, score, w, cap=.10):
    fpr, tpr, _ = roc_curve(y, score, sample_weight=w)
    ok = np.where(fpr <= cap + 1e-12)[0]
    return float(np.max(tpr[ok])) if len(ok) else 0.0


def metrics(y, score, w, thr):
    return (
        float(roc_auc_score(y, score, sample_weight=w)),
        float(average_precision_score(y, score, sample_weight=w)),
        float(
            f1_score(
                y,
                (score >= thr).astype(int),
                sample_weight=w,
                zero_division=0,
            )
        ),
        float(weighted_p_at_budget(y, score, w, .05)),
        float(weighted_r_at_fpr(y, score, w, .10)),
    )


def classify(base_name, j):
    model = j["model"]
    modal = j["modal"]
    label = j["label_col"]
    seed = int(j["seed"])

    if base_name == "core_v1":
        condition = f"{model}_{modal}"
        protocol = "neural_weighted_primary"
    elif base_name == "pure_graph_v1":
        condition = f"{model}_G6"
        protocol = "pure_graph"
    elif base_name == "label_sensitivity_v1":
        short = (
            "AONLY" if label == AONLY
            else "LOOSE" if label == LOOSE
            else "UNKNOWN"
        )
        condition = f"{short}_{model}_M11"
        protocol = "label_sensitivity"
    elif base_name == "audit_ablation_v1":
        condition = f"{model}_M11_NOAUDIT"
        protocol = "audit_ablation"
    elif base_name == "tabular_primary_v1":
        condition = f"TABUW_{model}_{modal}"
        protocol = "tabular_unweighted"
    elif base_name == "tabular_weighted_v1":
        condition = f"TABW_{model}_{modal}"
        protocol = "tabular_weighted"
    else:
        raise RuntimeError(base_name)

    return (
        condition,
        label,
        seed,
        float(j["val_threshold"]),
        protocol,
    )


def load_all(root):
    specs = [
        ("core_v1", 75),
        ("pure_graph_v1", 20),
        ("label_sensitivity_v1", 50),
        ("audit_ablation_v1", 25),
        ("tabular_primary_v1", 60),
        ("tabular_weighted_v1", 60),
    ]

    runs = []
    for base_name, expected in specs:
        base = root / "results" / base_name
        files = sorted(base.glob("*/result.json"))
        if len(files) != expected:
            raise RuntimeError(
                f"{base_name}: expected {expected}, got {len(files)}"
            )

        for rj in files:
            j = json.loads(rj.read_text())
            pred = rj.parent / "predictions.parquet"
            if not pred.is_file():
                raise RuntimeError(f"MISSING:{pred}")

            condition, label, seed, thr, protocol = classify(
                base_name, j
            )
            runs.append(
                (condition, label, seed, thr, protocol, pred)
            )

    if len(runs) != 290:
        raise RuntimeError(
            f"expected 290 runs, got {len(runs)}"
        )

    return runs


def build_universes(runs):
    universes = {}

    for label in [PRIMARY, AONLY, LOOSE]:
        subset = [x for x in runs if x[1] == label]
        master_keys = None
        master_y = None
        scores = {}
        thresholds = {}
        protocols = {}

        for condition, _, seed, thr, protocol, pred in subset:
            p = pd.read_parquet(
                pred,
                columns=[
                    "company_code",
                    "fiscal_year",
                    "y_true",
                    "score",
                ],
            )
            p["company_code"] = (
                p["company_code"].astype(str).str.zfill(6)
            )
            p["fiscal_year"] = p["fiscal_year"].astype(int)
            p = p.sort_values(
                ["company_code", "fiscal_year"]
            ).reset_index(drop=True)

            keys = list(
                zip(
                    p["company_code"],
                    p["fiscal_year"],
                )
            )
            y = p["y_true"].to_numpy(np.int8)

            if master_keys is None:
                master_keys = keys
                master_y = y
            else:
                if keys != master_keys:
                    raise RuntimeError(
                        f"KEY_MISMATCH:{pred}"
                    )
                if not np.array_equal(y, master_y):
                    raise RuntimeError(
                        f"LABEL_MISMATCH:{pred}"
                    )

            key = (condition, seed)
            if key in scores:
                raise RuntimeError(
                    f"DUPLICATE_CONDITION_SEED:{key}"
                )

            scores[key] = p["score"].to_numpy(
                np.float64
            )
            thresholds[key] = thr
            protocols[condition] = protocol

        keydf = pd.DataFrame(
            master_keys,
            columns=[
                "company_code",
                "fiscal_year",
            ],
        )
        cat = pd.Categorical(
            keydf["company_code"]
        )

        universes[label] = {
            "y": master_y,
            "scores": scores,
            "thresholds": thresholds,
            "protocols": protocols,
            "conditions": sorted(
                {k[0] for k in scores}
            ),
            "cluster_idx": cat.codes.astype(
                np.int32
            ),
            "n_clusters": len(cat.categories),
        }

    expected = {
        PRIMARY: (8435, 20),
        AONLY: (8434, 19),
        LOOSE: (8424, 31),
    }

    for label, (n, p) in expected.items():
        u = universes[label]
        got = (
            len(u["y"]),
            int(u["y"].sum()),
        )
        if got != (n, p):
            raise RuntimeError(
                f"{label}: expected {(n,p)}, got {got}"
            )

    return universes


def observed(universes):
    rows = []

    for label, u in universes.items():
        y = u["y"]
        w = np.ones(len(y), float)

        for condition in u["conditions"]:
            vals = []

            for seed in SEEDS:
                key = (condition, seed)
                if key not in u["scores"]:
                    raise RuntimeError(
                        f"MISSING_SEED:{label}:{condition}:{seed}"
                    )
                vals.append(
                    metrics(
                        y,
                        u["scores"][key],
                        w,
                        u["thresholds"][key],
                    )
                )

            a = np.asarray(vals)

            row = {
                "label": label,
                "condition": condition,
                "protocol": u["protocols"][condition],
                "n_seeds": 5,
                "test_n": len(y),
                "test_pos": int(y.sum()),
            }

            for i, name in enumerate(METRIC_NAMES):
                row[name + "_mean"] = float(
                    a[:, i].mean()
                )
                row[name + "_sd"] = float(
                    a[:, i].std(ddof=1)
                )

            rows.append(row)

    return pd.DataFrame(rows)


def _one_boot(task):
    b, base_seed = task
    out = {}

    for li, label in enumerate(
        [PRIMARY, AONLY, LOOSE]
    ):
        u = G["universes"][label]
        rng = np.random.default_rng(
            base_seed + b * 10 + li
        )

        while True:
            sampled = rng.integers(
                0,
                u["n_clusters"],
                size=u["n_clusters"],
            )
            counts = np.bincount(
                sampled,
                minlength=u["n_clusters"],
            )
            w = counts[
                u["cluster_idx"]
            ].astype(np.float64)

            y = u["y"]

            if (
                np.sum(w * y) > 0
                and np.sum(w * (1 - y)) > 0
            ):
                break

        for condition in u["conditions"]:
            vals = []

            for seed in SEEDS:
                vals.append(
                    metrics(
                        y,
                        u["scores"][
                            (condition, seed)
                        ],
                        w,
                        u["thresholds"][
                            (condition, seed)
                        ],
                    )
                )

            out[(label, condition)] = tuple(
                np.mean(
                    np.asarray(vals),
                    axis=0,
                )
            )

    return b, out


def seed_pair(universe, label, left, right, family):
    y = universe["y"]

    a = np.array(
        [
            roc_auc_score(
                y,
                universe["scores"][(left, s)],
            )
            for s in SEEDS
        ]
    )
    b = np.array(
        [
            roc_auc_score(
                y,
                universe["scores"][(right, s)],
            )
            for s in SEEDS
        ]
    )

    d = a - b

    return {
        "family": family,
        "label": label,
        "left": left,
        "right": right,
        "mean_delta_auc": float(d.mean()),
        "sd_delta_auc": float(d.std(ddof=1)),
        "paired_t_p": float(
            stats.ttest_rel(a, b).pvalue
        ),
        "exact_signflip_p": float(
            exact_signflip_p(d)
        ),
        "positive_seed_differences": int(
            np.sum(d > 0)
        ),
        "n_pairs": 5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=(
            Path.home()
            / "cyq/peerj_141707_rerun_v1"
        ),
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=2000,
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=141707,
    )
    args = ap.parse_args()

    root = args.root
    out = (
        root
        / "meta/posthoc_stats_v3"
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = load_all(root)
    universes = build_universes(runs)

    obs = observed(universes)
    obs.to_csv(
        out
        / "observed_condition_metrics.csv",
        index=False,
    )

    global G
    G = {
        "universes": universes
    }

    conditions = [
        (label, condition)
        for label, u in universes.items()
        for condition in u["conditions"]
    ]

    store = {
        k: np.empty(
            (args.n_boot, 5),
            dtype=np.float64,
        )
        for k in conditions
    }

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.jobs
    ) as pool:
        tasks = [
            (i, args.seed)
            for i in range(args.n_boot)
        ]

        for done, (b, res) in enumerate(
            pool.imap_unordered(
                _one_boot,
                tasks,
                chunksize=1,
            ),
            1,
        ):
            for k, v in res.items():
                store[k][b, :] = v

            if (
                done == 1
                or done % 50 == 0
            ):
                print(
                    f"bootstrap complete "
                    f"{done}/{args.n_boot}",
                    flush=True,
                )

    obsmap = {
        (r.label, r.condition): r
        for r in obs.itertuples(
            index=False
        )
    }

    # Per-condition company-cluster bootstrap CIs.
    ci_rows = []

    for key, a in store.items():
        label, condition = key
        r = obsmap[key]

        row = {
            "label": label,
            "condition": condition,
            "protocol": r.protocol,
            "n_boot": args.n_boot,
            "cluster_unit": "company_code",
            "test_n": r.test_n,
            "test_pos": r.test_pos,
            "unique_company_clusters":
                universes[label][
                    "n_clusters"
                ],
        }

        for i, name in enumerate(
            METRIC_NAMES
        ):
            row[name + "_mean"] = float(
                getattr(
                    r,
                    name + "_mean",
                )
            )
            row[name + "_ci_lo"] = float(
                np.quantile(
                    a[:, i],
                    .025,
                )
            )
            row[name + "_ci_hi"] = float(
                np.quantile(
                    a[:, i],
                    .975,
                )
            )

        ci_rows.append(row)

    pd.DataFrame(
        ci_rows
    ).to_csv(
        out
        / "condition_company_cluster_bootstrap_ci.csv",
        index=False,
    )

    # Targeted paired comparisons using the same company draw.
    deltas = []
    seedstats = []

    def add_pair(
        family,
        label,
        left,
        right,
        extra=None,
        do_seed_stats=False,
    ):
        if (
            (label, left) not in store
            or (label, right) not in store
        ):
            raise RuntimeError(
                f"MISSING_PAIR:{label}:{left}:{right}"
            )

        row = {
            "family": family,
            "label": label,
            "left": left,
            "right": right,
        }
        if extra:
            row.update(extra)

        for i, name in enumerate(
            METRIC_NAMES
        ):
            arr = (
                store[(label, left)][:, i]
                - store[(label, right)][:, i]
            )
            observed_delta = (
                getattr(
                    obsmap[(label, left)],
                    name + "_mean",
                )
                - getattr(
                    obsmap[(label, right)],
                    name + "_mean",
                )
            )

            row[
                name + "_observed_delta"
            ] = float(observed_delta)
            row[
                name + "_bootstrap_ci_lo"
            ] = float(
                np.quantile(arr, .025)
            )
            row[
                name + "_bootstrap_ci_hi"
            ] = float(
                np.quantile(arr, .975)
            )
            row[
                name + "_bootstrap_prob_delta_le_0"
            ] = float(
                np.mean(arr <= 0)
            )

        deltas.append(row)

        if do_seed_stats:
            seedstats.append(
                seed_pair(
                    universes[label],
                    label,
                    left,
                    right,
                    family,
                )
            )

    # A. Primary neural/GNN architecture comparisons.
    for modal in ["M5", "M10", "M11"]:
        for gnn in [
            "GCN",
            "GAT",
            "SAGE",
            "RGCN",
        ]:
            add_pair(
                "architecture",
                PRIMARY,
                f"MLP_{modal}",
                f"{gnn}_{modal}",
                {"modal": modal},
                do_seed_stats=True,
            )

    # B. Primary neural/GNN modality comparisons.
    for model in [
        "MLP",
        "GCN",
        "GAT",
        "SAGE",
        "RGCN",
    ]:
        for left_modal, right_modal in [
            ("M10", "M5"),
            ("M11", "M5"),
            ("M11", "M10"),
        ]:
            add_pair(
                "neural_modality",
                PRIMARY,
                f"{model}_{left_modal}",
                f"{model}_{right_modal}",
                {"model": model},
                do_seed_stats=True,
            )

        add_pair(
            "audit_ablation",
            PRIMARY,
            f"{model}_M11",
            f"{model}_M11_NOAUDIT",
            {"model": model},
            do_seed_stats=True,
        )

    # C. Label robustness architecture checks.
    for label, short in [
        (AONLY, "AONLY"),
        (LOOSE, "LOOSE"),
    ]:
        for gnn in [
            "GCN",
            "GAT",
            "SAGE",
            "RGCN",
        ]:
            add_pair(
                "label_robustness_architecture",
                label,
                f"{short}_MLP_M11",
                f"{short}_{gnn}_M11",
                {"modal": "M11"},
                do_seed_stats=True,
            )

    # D. Matched-weight cross-family comparisons:
    #    MLP versus four weighted tabular learners, same modality.
    for modal in ["M5", "M10", "M11"]:
        for tabular in [
            "Lasso",
            "Ridge",
            "RandomForest",
            "XGBoost",
        ]:
            add_pair(
                "matched_weight_cross_family",
                PRIMARY,
                f"MLP_{modal}",
                f"TABW_{tabular}_{modal}",
                {
                    "modal": modal,
                    "tabular_model": tabular,
                },
                do_seed_stats=False,
            )

    # E. Weighted tabular modality comparisons.
    for model in [
        "Lasso",
        "Ridge",
        "RandomForest",
        "XGBoost",
    ]:
        for left_modal, right_modal in [
            ("M10", "M5"),
            ("M11", "M5"),
            ("M11", "M10"),
        ]:
            add_pair(
                "weighted_tabular_modality",
                PRIMARY,
                f"TABW_{model}_{left_modal}",
                f"TABW_{model}_{right_modal}",
                {"model": model},
                do_seed_stats=False,
            )

    # F. Unweighted tabular modality comparisons.
    for model in [
        "Lasso",
        "Ridge",
        "RandomForest",
        "XGBoost",
    ]:
        for left_modal, right_modal in [
            ("M10", "M5"),
            ("M11", "M5"),
            ("M11", "M10"),
        ]:
            add_pair(
                "unweighted_tabular_modality",
                PRIMARY,
                f"TABUW_{model}_{left_modal}",
                f"TABUW_{model}_{right_modal}",
                {"model": model},
                do_seed_stats=False,
            )

    # G. Direct effect of class weighting in each tabular learner/modal.
    for model in [
        "Lasso",
        "Ridge",
        "RandomForest",
        "XGBoost",
    ]:
        for modal in [
            "M5",
            "M10",
            "M11",
        ]:
            add_pair(
                "tabular_weighting_effect",
                PRIMARY,
                f"TABW_{model}_{modal}",
                f"TABUW_{model}_{modal}",
                {
                    "model": model,
                    "modal": modal,
                },
                do_seed_stats=False,
            )

    pd.DataFrame(
        deltas
    ).to_csv(
        out
        / "paired_metric_delta_company_cluster_bootstrap_ci.csv",
        index=False,
    )

    pd.DataFrame(
        seedstats
    ).to_csv(
        out
        / "seed_paired_auc_stats.csv",
        index=False,
    )

    # Compact primary summary for manuscript drafting.
    primary_ci = pd.DataFrame(ci_rows)
    keep = (
        primary_ci["label"].eq(PRIMARY)
        & primary_ci["protocol"].isin(
            [
                "neural_weighted_primary",
                "tabular_weighted",
                "tabular_unweighted",
            ]
        )
    )
    primary_ci.loc[keep].sort_values(
        [
            "protocol",
            "roc_auc_mean",
        ],
        ascending=[
            True,
            False,
        ],
    ).to_csv(
        out
        / "primary_learner_summary_for_manuscript.csv",
        index=False,
    )

    report = [
        "PeerJ 141707 | Comprehensive post-hoc company-cluster bootstrap v3",
        "formal_prediction_runs_loaded=290",
        f"n_boot={args.n_boot}",
        f"jobs={args.jobs}",
        f"rng_seed={args.seed}",
        f"primary_test_n={len(universes[PRIMARY]['y'])}",
        f"primary_test_pos={int(universes[PRIMARY]['y'].sum())}",
        f"aonly_test_n={len(universes[AONLY]['y'])}",
        f"aonly_test_pos={int(universes[AONLY]['y'].sum())}",
        f"loose_test_n={len(universes[LOOSE]['y'])}",
        f"loose_test_pos={int(universes[LOOSE]['y'].sum())}",
        f"primary_company_clusters={universes[PRIMARY]['n_clusters']}",
        f"aonly_company_clusters={universes[AONLY]['n_clusters']}",
        f"loose_company_clusters={universes[LOOSE]['n_clusters']}",
        "same_cluster_draw_across_models_and_seeds_within_label=true",
        "paired_delta_metrics=roc_auc,ap,f1,p_at_5pct,r_at_10fpr",
        "matched_weight_cross_family_comparisons=true",
        "unweighted_tabular_retained_as_robustness=true",
        "raw_ids_or_scores_written=false",
        "gpu_used=false",
        "training_run=false",
        "STATUS: COMPREHENSIVE_POSTHOC_CLUSTER_BOOTSTRAP_V3_COMPLETE",
    ]

    (
        out
        / "posthoc_comprehensive_report.txt"
    ).write_text(
        "\n".join(report) + "\n"
    )

    print(
        "\n".join(report)
    )


if __name__ == "__main__":
    main()
