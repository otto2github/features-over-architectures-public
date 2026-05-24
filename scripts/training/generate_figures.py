"""
generate_figures.py
====================

Paper figure generation: produces four figures (PNG/PDF) from the result JSONs.

Output (figures/ directory):
  fig_modal_progression.{png,pdf}    # M5 → M10 → M11 progression curves (5 models)
  fig_ablation_bars.{png,pdf}         # single-modality ablation bar chart
  fig_robustness.{png,pdf}            # three-labelrobustness
  fig_train_curves.{png,pdf}          # training curves (demonstrating GNN early stopping)

Usage:
  python generate_figures.py --project-root /path/to/project [--out-dir /path/to/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paper-grade matplotlib settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results(results_dir: Path):
    """Load all relevant JSONs from results_dir."""
    out = {}
    for p in results_dir.glob("gnn_baseline_v1_1_*.json"):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception as e:
            print(f"reading {p} failed: {e}")
    return out


def fig_modal_progression(data, out_dir: Path):
    """Figure (modality progression): M5 → M10 → M11 three-stage modality improvement (5 models, 5 line-plots)."""
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    if not main:
        print("⚠ missing main_v2_strict data,skip fig_7_1")
        return

    by_model = {}
    for r in main.get("summary", []):
        by_model.setdefault(r["model"], {})[r["modal"]] = r["test_auc"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    modals = ["M5", "M10", "M11"]
    x_labels = ["M5\n(v0.7, 104d)", "M10\n(+audit/pld/ctrl, 122d)", "M11\n(+rpt, 129d)"]
    x = np.arange(len(modals))
    colors = {"MLP": "#333333", "GCN": "#1f77b4", "GAT": "#ff7f0e",
              "SAGE": "#2ca02c", "RGCN": "#d62728"}
    markers = {"MLP": "o", "GCN": "s", "GAT": "^", "SAGE": "D", "RGCN": "v"}

    for model in ["MLP", "GCN", "GAT", "SAGE", "RGCN"]:
        if model not in by_model:
            continue
        ys = [by_model[model].get(m, np.nan) for m in modals]
        ax.plot(x, ys, label=model, color=colors[model],
                marker=markers[model], markersize=8, linewidth=1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Modal Configuration")
    ax.set_ylabel("Test AUC")
    ax.set_title("Figure (modality progression). Test AUC across Modal Progression (M5 → M10 → M11)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    ax.set_ylim(0.64, 0.74)

    for ext in ["png", "pdf"]:
        path = out_dir / f"fig_modal_progression.{ext}"
        fig.savefig(path)
        print(f"  ✓ {path}")
    plt.close(fig)


def fig_ablation_bars(data, out_dir: Path):
    """Figure (ablation bars): M6/M7/M8/M9 single-modality contribution vs M5 (5-model grouped bar chart)."""
    abl = data.get("gnn_baseline_v1_1_ablation_strict_full", {})
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    if not abl or not main:
        print("⚠ missing ablation or main data, skipping (ablation) figure")
        return

    # M5 baseline per model
    m5_baseline = {r["model"]: r["test_auc"] for r in main.get("summary", [])
                    if r["modal"] == "M5"}

    # M6-M9 per model
    by_model = {}
    for r in abl.get("summary", []):
        by_model.setdefault(r["model"], {})[r["modal"]] = r["test_auc"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    models = ["MLP", "GCN", "GAT", "SAGE", "RGCN"]
    modals_added = ["M6\n(+audit)", "M7\n(+pld)", "M8\n(+ctrl)", "M9\n(+rpt)"]
    modal_keys = ["M6", "M7", "M8", "M9"]

    bar_width = 0.16
    x = np.arange(len(modals_added))
    colors = {"MLP": "#333333", "GCN": "#1f77b4", "GAT": "#ff7f0e",
              "SAGE": "#2ca02c", "RGCN": "#d62728"}

    for i, model in enumerate(models):
        if model not in by_model:
            continue
        baseline = m5_baseline.get(model, 0)
        # ΔAUC vs M5
        deltas = [by_model[model].get(m, np.nan) - baseline for m in modal_keys]
        ax.bar(x + (i - 2) * bar_width, deltas, bar_width,
               label=f"{model} (M5={baseline:.4f})",
               color=colors[model], alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(modals_added)
    ax.set_xlabel("Modal Added (over M5 baseline)")
    ax.set_ylabel("Δ Test AUC (relative to M5)")
    ax.set_title("Figure (ablation). Single-modal Ablation (Δ AUC vs M5 baseline)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=9)

    for ext in ["png", "pdf"]:
        path = out_dir / f"fig_ablation_bars.{ext}"
        fig.savefig(path)
        print(f"  ✓ {path}")
    plt.close(fig)


def fig_robustness(data, out_dir: Path):
    """Figure (robustness): AUC robustness of M11 across three label protocols."""
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    v07 = data.get("gnn_baseline_v1_1_robust_v07", {})
    loose = data.get("gnn_baseline_v1_1_robust_loose", {})
    if not (main and v07 and loose):
        print("⚠ missing robustness data,skip fig_7_3")
        return

    def m11_aucs(d):
        return {r["model"]: r["test_auc"]
                for r in d.get("summary", []) if r["modal"] == "M11"}

    strict_dict = m11_aucs(main)
    v07_dict = m11_aucs(v07)
    loose_dict = m11_aucs(loose)

    models = ["MLP", "GCN", "GAT", "SAGE", "RGCN"]
    labels = ["fraud_v07\n(akshare)", "fraud_v08_strict\n(primary label)", "fraud_v08_loose\n(loose)"]
    label_keys = [v07_dict, strict_dict, loose_dict]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_width = 0.16
    x = np.arange(len(labels))
    colors = {"MLP": "#333333", "GCN": "#1f77b4", "GAT": "#ff7f0e",
              "SAGE": "#2ca02c", "RGCN": "#d62728"}

    for i, model in enumerate(models):
        ys = [d.get(model, np.nan) for d in label_keys]
        ax.bar(x + (i - 2) * bar_width, ys, bar_width,
               label=model, color=colors[model], alpha=0.85,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Label Definition")
    ax.set_ylabel("Test AUC")
    ax.set_title("Figure (robustness). Robustness to Label Definition (M11)")
    ax.set_ylim(0.55, 0.75)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)

    for ext in ["png", "pdf"]:
        path = out_dir / f"fig_robustness.{ext}"
        fig.savefig(path)
        print(f"  ✓ {path}")
    plt.close(fig)


def fig_7_4_pure_graph_diagnostic(data, out_dir: Path):
    """Figure (training curves): Pure-graph diagnostic G6 + Big-GNN comparison."""
    pure = data.get("gnn_baseline_v1_1_pure_graph", {})
    big = data.get("gnn_baseline_v1_1_big_gnn", {})
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    if not (pure and main):
        print("⚠ missing pure_graph data,skip fig_7_4")
        return

    pure_dict = {r["model"]: r["test_auc"] for r in pure.get("summary", [])}
    main_m11 = {r["model"]: r["test_auc"]
                for r in main.get("summary", []) if r["modal"] == "M11"}
    big_dict = {r["model"]: r["test_auc"] for r in (big.get("summary", []) if big else [])}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = ["GCN", "GAT", "SAGE", "RGCN"]
    x = np.arange(len(models))
    bar_width = 0.27
    colors = ["#888888", "#1f77b4", "#d62728"]

    pure_ys = [pure_dict.get(m, np.nan) for m in models]
    main_ys = [main_m11.get(m, np.nan) for m in models]
    big_ys = [big_dict.get(m, np.nan) if big_dict else np.nan for m in models]

    ax.bar(x - bar_width, pure_ys, bar_width,
           label="Pure Graph G6 (no features)", color=colors[0],
           edgecolor="black", linewidth=0.5)
    ax.bar(x, main_ys, bar_width,
           label="M11, hidden=64 (main)", color=colors[1],
           edgecolor="black", linewidth=0.5)
    if any(not np.isnan(y) for y in big_ys):
        ax.bar(x + bar_width, big_ys, bar_width,
               label="M11, hidden=256 (large)", color=colors[2],
               edgecolor="black", linewidth=0.5)

    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.0, label="Random")
    ax.axhline(main_m11.get("MLP", 0.7144), color="black", linestyle="--",
               linewidth=1.0, label=f"MLP-M11 baseline ({main_m11.get('MLP', 0.7144):.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("GNN Model")
    ax.set_ylabel("Test AUC")
    ax.set_title("Figure 7-4. Pure Graph Diagnostic vs M11 Performance")
    ax.set_ylim(0.4, 0.78)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    for ext in ["png", "pdf"]:
        path = out_dir / f"fig_7_4_pure_graph_diagnostic.{ext}"
        fig.savefig(path)
        print(f"  ✓ {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    proj = args.project_root or (Path.home() / "code" / "foa_project")
    results_dir = proj / "data" / "interim" / "p7_gnn_results"
    out_dir = args.out_dir or (proj / "data" / "interim" / "p7_gnn_results" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading results from: {results_dir}")
    data = load_results(results_dir)
    print(f"  loaded {len(data)} JSON files")
    print(f"Output figures to: {out_dir}")
    print()

    print("generated Fig (modality progression) (Modal Progression) ...")
    fig_modal_progression(data, out_dir)
    print()
    print("generated Fig (ablation) (Ablation Bars) ...")
    fig_ablation_bars(data, out_dir)
    print()
    print("generated Fig (robustness) (Robustness) ...")
    fig_robustness(data, out_dir)
    print()
    print("generated Fig 7-4 (Pure Graph Diagnostic) ...")
    fig_7_4_pure_graph_diagnostic(data, out_dir)
    print()
    print("✓ All done")


if __name__ == "__main__":
    main()
