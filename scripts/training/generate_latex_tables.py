"""
generate_latex_tables.py
=========================

Paper LaTeX table generation (booktabs style).

Output (tables/ directory):
  table_main.tex          # main benchmark (15 experiments × 5 metrics)
  table_delong.tex       # DeLong significance test
  table_ablation.tex      # single-modality ablation
  table_pure_graph.tex    # pure-graph diagnostic
  table_robustness.tex    # three-labelrobustness

Usage:
  python generate_latex_tables.py --project-root /path/to/project
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path):
    out = {}
    for p in results_dir.glob("gnn_baseline_v1_1_*.json"):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception as e:
            print(f"reading {p} failed: {e}")
    return out


def write(path: Path, content: str):
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ {path}")


def fmt(x, prec=4):
    if x is None or x != x:
        return "—"
    return f"{x:.{prec}f}"


def fmt_p(p):
    if p is None or p != p:
        return "—"
    if p < 0.001:
        return f"{p:.4f}\\textsuperscript{{***}}"
    if p < 0.01:
        return f"{p:.4f}\\textsuperscript{{**}}"
    if p < 0.05:
        return f"{p:.4f}\\textsuperscript{{*}}"
    return f"{p:.4f}"


def table_main(data, out_dir):
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    if not main:
        return
    rows = main.get("summary", [])
    by_modal = {}
    for r in rows:
        by_modal.setdefault(r["modal"], []).append(r)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Table (modality progression). Main Results: Test Set Performance across Models and Modal Configurations}}")
    lines.append(r"\label{tab:7_1_main}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Modal & Model & Best Ep. & Val AUC & \textbf{Test AUC} & AP & F1@best & P@5\% \\")
    lines.append(r"\midrule")

    for modal in ["M5", "M10", "M11"]:
        if modal not in by_modal:
            continue
        for i, r in enumerate(by_modal[modal]):
            modal_cell = modal if i == 0 else ""
            lines.append(f"{modal_cell} & {r['model']} & {r['best_epoch']} & "
                          f"{fmt(r['val_auc'])} & \\textbf{{{fmt(r['test_auc'])}}} & "
                          f"{fmt(r['ap'])} & {fmt(r['f1_best'])} & {fmt(r['p_at_5pct'])} \\\\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item M5: v0.7 all features (104 -dim, feat\_+fin\_+fini\_); ")
    lines.append(r"M10: v1.0 (122 -dim, M5+audit+pld+ctrl); ")
    lines.append(r"M11: v1.1 (129 -dim, M10+rpt). ")
    lines.append(r"Label: \texttt{fraud\_v08\_strict} (CSMAR P2501-P2503+P2507).")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    write(out_dir / "table_main.tex", "\n".join(lines))


def table_delong(data, out_dir):
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    if not main:
        return
    delong = main.get("delong", [])
    if not delong:
        return

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Table (modality progression)b. DeLong Test for Modal Progression}}")
    lines.append(r"\label{tab:7_1b_delong}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & $\Delta$AUC & $z$ & $p$-value \\")
    lines.append(r"\midrule")

    # Modal progression
    lines.append(r"\multicolumn{4}{l}{\textit{Panel A: Modal Progression (M5 $\to$ M10 $\to$ M11)}} \\")
    for d in delong:
        comp = d["comparison"]
        if "vs" not in comp or any(prefix in comp for prefix in ["MLP)", "MLP "]):
            continue
        if "M5" not in comp and "M10" not in comp and "M11" not in comp:
            continue
        # modality comparison only, excluding GNN vs MLP
        if " vs MLP " in comp or "vs MLP (" in comp:
            continue
        lines.append(f"{comp.replace('_', '\\_')} & {d['delta']:+.4f} & {d['z']:+.3f} & {fmt_p(d['p'])} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Panel B: GNN vs MLP (Same Modal)}} \\")
    for d in delong:
        comp = d["comparison"]
        if "vs MLP" not in comp:
            continue
        lines.append(f"{comp.replace('_', '\\_')} & {d['delta']:+.4f} & {d['z']:+.3f} & {fmt_p(d['p'])} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. ")
    lines.append(r"DeLong's paired AUC test (Sun \& Xu, 2014).")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    write(out_dir / "table_delong.tex", "\n".join(lines))


def table_ablation(data, out_dir):
    abl = data.get("gnn_baseline_v1_1_ablation_strict_full", {})
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    if not abl or not main:
        return

    m5 = {r["model"]: r["test_auc"] for r in main.get("summary", [])
          if r["modal"] == "M5"}
    by_model = {}
    for r in abl.get("summary", []):
        by_model.setdefault(r["model"], {})[r["modal"]] = r["test_auc"]

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Table (ablation). Single-modal Ablation: $\Delta$AUC vs M5 Baseline}}")
    lines.append(r"\label{tab:7_2_ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & M5 (base) & M6 (+audit) & M7 (+pld) & M8 (+ctrl) & M9 (+rpt) \\")
    lines.append(r"\midrule")

    for model in ["MLP", "GCN", "GAT", "SAGE", "RGCN"]:
        if model not in by_model:
            continue
        b = m5.get(model, 0)
        cells = [f"{b:.4f}"]
        for ma in ["M6", "M7", "M8", "M9"]:
            v = by_model[model].get(ma, None)
            if v is None:
                cells.append("—")
            else:
                delta = v - b
                cells.append(f"{v:.4f} ({delta:+.4f})")
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item Each modality is added on top of M5 (v0.7); the parenthetical value is $\Delta$AUC relative to M5.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    write(out_dir / "table_ablation.tex", "\n".join(lines))


def table_pure_graph(data, out_dir):
    pure = data.get("gnn_baseline_v1_1_pure_graph", {})
    if not pure:
        return

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Table (robustness). Pure Graph Diagnostic (G6: All-ones Node Features)}}")
    lines.append(r"\label{tab:7_3_pure_graph}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Test AUC & F1@best & Interpretation \\")
    lines.append(r"\midrule")

    for r in pure.get("summary", []):
        auc = r["test_auc"]
        if auc < 0.55:
            interp = r"\textit{No signal}"
        elif auc < 0.60:
            interp = r"\textit{Weak signal}"
        else:
            interp = r"\textit{Has signal}"
        lines.append(f"{r['model']} & {fmt(auc)} & {fmt(r['f1_best'])} & {interp} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item G6 modality uses all-ones node features so the model relies on graph topology only. ")
    lines.append(r"AUC $<$ 0.55: no graph signal; 0.55--0.60: weak signal; $>$ 0.60: signal present.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    write(out_dir / "table_pure_graph.tex", "\n".join(lines))


def table_robustness(data, out_dir):
    main = data.get("gnn_baseline_v1_1_main_v2_strict", {})
    v07 = data.get("gnn_baseline_v1_1_robust_v07", {})
    loose = data.get("gnn_baseline_v1_1_robust_loose", {})
    if not (main and v07 and loose):
        return

    def m11_dict(d):
        return {r["model"]: r["test_auc"] for r in d.get("summary", [])
                if r["modal"] == "M11"}

    strict_d = m11_dict(main)
    v07_d = m11_dict(v07)
    loose_d = m11_dict(loose)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Table 7-5. Robustness to Label Definition (M11 only)}}")
    lines.append(r"\label{tab:7_5_robust}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & fraud\_v07 & fraud\_v08\_strict & fraud\_v08\_loose & Range \\")
    lines.append(r" & (akshare) & (\textbf{main}) & (relaxed) & \\")
    lines.append(r"\midrule")

    for m in ["MLP", "GCN", "GAT", "SAGE", "RGCN"]:
        v07v = v07_d.get(m, None)
        sv = strict_d.get(m, None)
        lv = loose_d.get(m, None)
        vals = [x for x in [v07v, sv, lv] if x is not None]
        rng = max(vals) - min(vals) if len(vals) > 1 else 0
        lines.append(f"{m} & {fmt(v07v)} & \\textbf{{{fmt(sv)}}} & {fmt(lv)} & {rng:.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item Range = max - min (across 3 labels); typically $<$ 0.03 is considered robust.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    write(out_dir / "table_robustness.tex", "\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    proj = args.project_root or (Path.home() / "code" / "foa_project")
    results_dir = proj / "data" / "interim" / "p7_gnn_results"
    out_dir = args.out_dir or (results_dir / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {results_dir}")
    data = load_results(results_dir)
    print(f"  Loaded {len(data)} JSON files\nOutput: {out_dir}\n")

    print("Table (modality progression) main benchmark ...")
    table_main(data, out_dir)
    print("DeLong significance table ...")
    table_delong(data, out_dir)
    print("Table (ablation) single-modality ablation ...")
    table_ablation(data, out_dir)
    print("Table (robustness) pure-graph diagnostic ...")
    table_pure_graph(data, out_dir)
    print("Table 7-5 robustness ...")
    table_robustness(data, out_dir)
    print("\n✓ All done")


if __name__ == "__main__":
    main()
