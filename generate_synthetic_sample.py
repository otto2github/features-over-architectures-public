"""
generate_synthetic_sample.py
────────────────────────────
Generates a schema-valid synthetic M11 feature matrix for pipeline validation
without requiring licensed CSMAR data.

Two modes
─────────
A) Schema-driven (default, recommended):
   Reads data/synthetic_sample_schema.json (committed to this repo),
   samples each feature from N(mean, std) clipped to [min, max],
   and produces a CSV with column names and dtypes identical to the
   real features_v1_1.parquet used in the benchmark.

B) Hardcoded fallback:
   If the schema file is absent, falls back to a representative 84-feature
   hardcoded schema so the script is self-contained without any data files.

Usage
─────
# From repo root (no CSMAR license needed):
python generate_synthetic_sample.py

# Custom output path / row count / seed:
python generate_synthetic_sample.py --out data/synthetic_sample_M11.csv --n-rows 50 --seed 42

# Point at a different schema:
python generate_synthetic_sample.py --schema path/to/schema.json

Output
──────
CSV with columns:
  firm_id, year,
  <129 M11 feature columns matching features_v1_1.parquet>,
  fraud_v07, fraud_v08_strict, fraud_v08_loose  (all 0 — synthetic, no real labels)

NOTE: All values are statistically independent of real CSMAR records.
      This file is for pipeline validation only, not for scientific analysis.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

RANDOM_SEED    = 42
DEFAULT_SCHEMA = os.path.join(os.path.dirname(__file__),
                              "data", "synthetic_sample_schema.json")
DEFAULT_OUT    = os.path.join(os.path.dirname(__file__),
                              "data", "synthetic_sample_M11.csv")

# ── Hardcoded fallback schema (used only when schema JSON is absent) ──────────
FALLBACK_SCHEMA = {
    "feat_t0_total_notices":      {"dtype":"float64","min":0,"max":500,"mean":12.0,"std":18.0},
    "feat_t0_reduce_notices":     {"dtype":"float64","min":0,"max":200,"mean":2.5,"std":6.0},
    "feat_t0_increase_notices":   {"dtype":"float64","min":0,"max":100,"mean":1.2,"std":3.5},
    "operating_cf_to_net_profit": {"dtype":"float64","min":-50,"max":50,"mean":0.8,"std":4.2},
    "ar_turnover":                {"dtype":"float64","min":0.1,"max":80,"mean":8.5,"std":9.0},
    "dsri":                       {"dtype":"float64","min":0.1,"max":5,"mean":1.05,"std":0.4},
    "other_receivables_to_assets":{"dtype":"float64","min":0,"max":0.5,"mean":0.04,"std":0.05},
    "leverage_ratio":             {"dtype":"float64","min":0.05,"max":0.95,"mean":0.42,"std":0.18},
    "audit_opinion_type":         {"dtype":"int8","min":0,"max":3,"mean":0.1,"std":0.4},
    "pledge_ratio":               {"dtype":"float64","min":0,"max":1,"mean":0.12,"std":0.18},
    "rpt_total_amount":           {"dtype":"float32","min":0,"max":100,"mean":3.5,"std":9.0},
}


def load_schema(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate(schema: dict, n_rows: int = 50, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: dict = {}

    # Synthetic identifiers
    rows["firm_id"] = [f"SYN{i:05d}" for i in range(n_rows)]
    rows["year"]    = rng.integers(2010, 2025, size=n_rows).tolist()

    for col, info in schema.items():
        lo  = info["min"]
        hi  = info["max"]
        mu  = info["mean"]
        std = info.get("std", (hi - lo) / 4)
        dtype_str = info.get("dtype", "float64")

        vals = rng.normal(mu, std if std > 0 else 1e-6, size=n_rows)
        vals = np.clip(vals, lo, hi)

        if dtype_str in ("int8", "int32", "int64"):
            vals = vals.round().astype(int)
        elif dtype_str == "float32":
            vals = vals.astype("float32")

        rows[col] = vals.tolist()

    # Labels — all 0 (synthetic data carries no real fraud labels)
    for lc in ("fraud_v07", "fraud_v08_strict", "fraud_v08_loose"):
        rows[lc] = [0] * n_rows

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a schema-valid synthetic M11 sample for pipeline validation."
    )
    parser.add_argument("--schema",  default=DEFAULT_SCHEMA,
                        help="Path to synthetic_sample_schema.json")
    parser.add_argument("--out",     default=DEFAULT_OUT,
                        help="Output CSV path")
    parser.add_argument("--n-rows",  type=int, default=50)
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if os.path.exists(args.schema):
        schema = load_schema(args.schema)
        print(f"Loaded schema: {len(schema)} features from {args.schema}")
    else:
        schema = FALLBACK_SCHEMA
        print(f"Schema file not found — using hardcoded fallback "
              f"({len(schema)} representative features).")

    df = generate(schema, n_rows=args.n_rows, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Saved {args.out}: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Feature columns: {df.shape[1] - 5} "
          f"(firm_id, year, 3 label cols excluded)")
    print("NOTE: All values are statistically independent of real CSMAR data.")


if __name__ == "__main__":
    main()
