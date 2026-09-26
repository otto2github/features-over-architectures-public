#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import formal_rerun_core as frc

ORIGINAL_SELECT_COLS = frc.select_cols


def select_cols_ablation(df, modal):
    if modal != "M11_NOAUDIT":
        return ORIGINAL_SELECT_COLS(df, modal)

    prefixes = (
        "feat_",
        "fin_",
        "fini_",
        "audit_",
        "pld_",
        "ctrl_",
        "rpt_",
    )

    cols = sorted(
        c for c in df.columns
        if any(c.startswith(p) for p in prefixes)
        and not c.startswith("audit_")
    )

    if len(cols) != 124:
        raise RuntimeError(
            f"M11_NOAUDIT_DIM_MISMATCH: got={len(cols)} expected=124"
        )

    audit_cols = sorted(
        c for c in df.columns
        if c.startswith("audit_")
    )

    if len(audit_cols) != 5:
        raise RuntimeError(
            f"AUDIT_BLOCK_DIM_MISMATCH: got={len(audit_cols)} expected=5"
        )

    return cols


# Patch only this independent ablation process.
frc.select_cols = select_cols_ablation


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument(
        "--model",
        choices=["MLP", "GCN", "GAT", "SAGE", "RGCN"],
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        choices=[42, 123, 456, 789, 1024],
        required=True,
    )

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    args.modal = "M11_NOAUDIT"
    args.label_col = "label_v1_strict_ab_primary"
    args.no_test = False

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = frc.load_joined(
        args.data_dir,
        args.label_col,
    )

    cols = select_cols_ablation(
        df,
        "M11_NOAUDIT",
    )

    if len(cols) != 124:
        raise RuntimeError("M11_NOAUDIT != 124")

    pw, neg, pos = frc.dynamic_pos_weight(df)

    if (neg, pos) != (23852, 254):
        raise RuntimeError(
            f"PRIMARY_TRAIN_COUNTS_MISMATCH: neg={neg} pos={pos}"
        )

    started = time.time()

    if args.model == "MLP":
        result = frc.run_mlp(
            df,
            "M11_NOAUDIT",
            args.seed,
            args,
            args.out_dir,
        )
    else:
        result = frc.run_gnn(
            df,
            args.model,
            "M11_NOAUDIT",
            args.seed,
            args,
            args.out_dir,
            args.data_dir,
        )

    result.update({
        "label_col": args.label_col,
        "feature_condition": "M11_NOAUDIT",
        "feature_dim": 124,
        "removed_block": "audit_*",
        "removed_feature_count": 5,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "device": args.device,
        "runtime_seconds": time.time() - started,
        "protocol_note":
            "Identical to frozen primary protocol except "
            "the five audit_* columns are excluded from M11.",
    })

    (
        args.out_dir / "result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
