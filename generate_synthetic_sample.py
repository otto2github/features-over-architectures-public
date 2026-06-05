"""
generate_synthetic_sample.py
────────────────────────────
Generates a schema-valid synthetic feature matrix that can be fed directly into
the benchmark pipeline without requiring a licensed CSMAR data extract.

The synthetic data is statistically independent of any real CSMAR records. All
values are sampled from distributions calibrated to publicly reported ranges for
Chinese A-share financial data; they do not constitute real financial information
about any firm.

Usage (run on any machine with the feature_dictionary.json available):
    python generate_synthetic_sample.py \
        --feature-dict  /path/to/feature_dictionary.json \
        --n-rows        50 \
        --out           data/synthetic_sample_M11.csv \
        --seed          42

If feature_dictionary.json is not available (e.g. on a machine without the
CSMAR extract), the script falls back to the hardcoded M11 representative schema
defined in REPRESENTATIVE_FEATURES below.

Output CSV schema:
    - One row per synthetic firm-year observation.
    - Columns match M11 feature names exactly (compatible with prepare_node_features_v1_1.py).
    - 'fraud_v08_strict' label column added (0 for all rows; users can set positives manually).
    - 'firm_id' and 'year' columns added for traceability.
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

RANDOM_SEED = 42

# ── Representative M11 feature schema (fallback when feature_dictionary.json unavailable) ──
# Structure: {column_name: (dtype, low, high)}  for numeric uniform sampling.
# For ratios bounded to [0,1] the range is (0.0, 1.0); for financial magnitudes
# the range reflects typical Chinese A-share annual-report values (in 亿 RMB units
# for monetary, dimensionless for ratios/scores).

REPRESENTATIVE_FEATURES = {
    # ── M5: insider trading (17) ──
    "insider_buy_ratio":      ("float32",  0.0,   0.15),
    "insider_sell_ratio":     ("float32",  0.0,   0.20),
    "insider_net_buy_ratio":  ("float32", -0.20,  0.15),
    "insider_trade_count":    ("float32",  0.0,  50.0),
    "director_buy_vol":       ("float32",  0.0,   5.0),
    "director_sell_vol":      ("float32",  0.0,   8.0),
    "supervisor_trade_flag":  ("float32",  0.0,   1.0),
    "senior_mgmt_trade_flag": ("float32",  0.0,   1.0),
    "insider_trade_month":    ("float32",  1.0,  12.0),
    "insider_trade_q4_flag":  ("float32",  0.0,   1.0),
    "director_net_shares":    ("float32", -2.0,   2.0),
    "supervisor_net_shares":  ("float32", -1.0,   1.0),
    "mgmt_net_shares":        ("float32", -1.5,   2.0),
    "insider_trade_size":     ("float32",  0.0,   3.0),
    "insider_pre_earnings":   ("float32",  0.0,   1.0),
    "insider_post_earnings":  ("float32",  0.0,   1.0),
    "insider_momentum_flag":  ("float32",  0.0,   1.0),
    # ── M5: raw financial-statement features (sample of 25; full set = 70) ──
    "total_revenue":          ("float32",  0.5, 500.0),
    "net_profit":             ("float32", -5.0,  50.0),
    "total_assets":           ("float32",  2.0, 800.0),
    "total_liabilities":      ("float32",  1.0, 600.0),
    "equity":                 ("float32",  0.5, 300.0),
    "operating_cashflow":     ("float32", -8.0,  60.0),
    "investing_cashflow":     ("float32",-20.0,  10.0),
    "financing_cashflow":     ("float32",-15.0,  20.0),
    "accounts_receivable":    ("float32",  0.0, 100.0),
    "inventory":              ("float32",  0.0,  80.0),
    "other_receivables":      ("float32",  0.0,  30.0),
    "fixed_assets":           ("float32",  0.0, 200.0),
    "intangible_assets":      ("float32",  0.0,  50.0),
    "short_term_debt":        ("float32",  0.0, 100.0),
    "long_term_debt":         ("float32",  0.0, 200.0),
    "gross_profit":           ("float32",  0.1, 150.0),
    "ebitda":                 ("float32", -3.0,  70.0),
    "retained_earnings":      ("float32", -5.0, 100.0),
    "minority_interest":      ("float32",  0.0,  20.0),
    "revenue_yoy":            ("float32", -0.5,   1.5),
    "profit_yoy":             ("float32", -2.0,   3.0),
    "asset_yoy":              ("float32", -0.3,   1.0),
    "capex":                  ("float32",  0.0,  30.0),
    "rd_expense":             ("float32",  0.0,  10.0),
    "admin_expense_ratio":    ("float32",  0.01,  0.3),
    # ── M5: derived financial features (17) ──
    "operating_cf_to_net_profit":     ("float32", -5.0,  5.0),   # 经营现金流/净利润 (top-5)
    "ar_turnover":                    ("float32",  1.0, 30.0),   # 应收账款周转率 (top-5)
    "dsri":                           ("float32",  0.3,  3.0),   # DSRI (top-5)
    "other_receivables_to_assets":    ("float32",  0.0,  0.3),   # 其他应收款/总资产 (top-5)
    "leverage_ratio":                 ("float32",  0.1,  0.9),   # 资产负债率 (top-5)
    "current_ratio":                  ("float32",  0.3,  5.0),
    "quick_ratio":                    ("float32",  0.2,  4.0),
    "asset_turnover":                 ("float32",  0.1,  3.0),
    "inventory_turnover":             ("float32",  0.5, 20.0),
    "gross_margin":                   ("float32",  0.0,  0.8),
    "operating_margin":               ("float32", -0.2,  0.5),
    "jones_accruals":                 ("float32", -0.2,  0.2),
    "m_score":                        ("float32", -3.5,  2.0),
    "roe":                            ("float32", -0.3,  0.4),
    "roa":                            ("float32", -0.15, 0.2),
    "earnings_quality":               ("float32", -1.0,  2.0),
    "growth_sustainability":          ("float32", -0.5,  1.5),
    # ── M10 adds: audit opinion (3) ──
    "audit_opinion_type":             ("float32",  0.0,  3.0),   # 0=unqualified,1=qualified,2=adverse,3=disclaimer
    "modified_opinion_flag":          ("float32",  0.0,  1.0),
    "audit_emphasis_flag":            ("float32",  0.0,  1.0),
    # ── M10 adds: share-pledge features (5) ──
    "pledge_ratio":                   ("float32",  0.0,  0.9),
    "controller_pledge_ratio":        ("float32",  0.0,  0.9),
    "pledge_count":                   ("float32",  0.0, 10.0),
    "pledge_release_ratio":           ("float32",  0.0,  1.0),
    "pledge_risk_flag":               ("float32",  0.0,  1.0),
    # ── M10 adds: controller features (10) ──
    "controller_type":                ("float32",  0.0,  4.0),   # 0=state,1=private,2=foreign,3=collective,4=other
    "controller_ownership_ratio":     ("float32",  0.1,  0.8),
    "controller_direct_ratio":        ("float32",  0.0,  0.6),
    "controller_indirect_ratio":      ("float32",  0.0,  0.4),
    "controller_chain_depth":         ("float32",  1.0,  5.0),
    "controlling_shareholder_change": ("float32",  0.0,  1.0),
    "parent_ownership_ratio":         ("float32",  0.0,  0.9),
    "dual_role_flag":                 ("float32",  0.0,  1.0),
    "board_independence_ratio":       ("float32",  0.3,  0.7),
    "supervisory_board_size":         ("float32",  3.0,  9.0),
    # ── M11 adds: RPT summaries (7) ──
    "rpt_total_amount":               ("float32",  0.0, 50.0),
    "rpt_fund_flow_amount":           ("float32",  0.0, 20.0),
    "rpt_guarantee_amount":           ("float32",  0.0, 15.0),
    "rpt_commercial_amount":          ("float32",  0.0, 20.0),
    "rpt_asset_amount":               ("float32",  0.0, 10.0),
    "rpt_count":                      ("float32",  0.0, 30.0),
    "rpt_to_assets_ratio":            ("float32",  0.0,  0.5),
}


def load_feature_dict(path: str) -> dict:
    """Load feature names and basic schema from feature_dictionary.json."""
    with open(path) as f:
        fd = json.load(f)
    # expected format: [{"name": "...", "dtype": "float32", "description": "..."}, ...]
    if isinstance(fd, list):
        return {item["name"]: ("float32", 0.0, 1.0) for item in fd}
    return {}


def generate_synthetic_sample(
    feature_schema: dict,
    n_rows: int = 50,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = {}

    # Firm/year identifiers (synthetic)
    rows["firm_id"] = [f"SYN{i:05d}" for i in range(n_rows)]
    rows["year"] = rng.integers(2010, 2025, size=n_rows).tolist()

    for col, (dtype, lo, hi) in feature_schema.items():
        if dtype in ("int32", "int64"):
            vals = rng.integers(int(lo), int(hi) + 1, size=n_rows).astype(dtype)
        else:
            vals = rng.uniform(lo, hi, size=n_rows).astype(dtype)
        rows[col] = vals

    # Label column — all zero (not-fraud) for synthetic data
    rows["fraud_v08_strict"] = np.zeros(n_rows, dtype="int8")

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic M11 sample for pipeline validation.")
    parser.add_argument("--feature-dict", default=None,
                        help="Path to feature_dictionary.json (optional; uses hardcoded schema if absent).")
    parser.add_argument("--n-rows", type=int, default=50,
                        help="Number of synthetic firm-year rows to generate (default: 50).")
    parser.add_argument("--out", default="data/synthetic_sample_M11.csv",
                        help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.feature_dict and os.path.exists(args.feature_dict):
        schema = load_feature_dict(args.feature_dict)
        print(f"Loaded {len(schema)} features from {args.feature_dict}")
    else:
        schema = REPRESENTATIVE_FEATURES
        print(f"Using hardcoded representative schema ({len(schema)} features).")

    df = generate_synthetic_sample(schema, n_rows=args.n_rows, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} synthetic rows × {len(df.columns)} columns → {args.out}")
    print(f"Columns: firm_id, year, {len(schema)} features, fraud_v08_strict")
    print("NOTE: All values are statistically independent of real CSMAR data.")


if __name__ == "__main__":
    main()
