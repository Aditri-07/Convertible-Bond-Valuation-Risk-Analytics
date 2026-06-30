"""
Analysis script: reproduces the divergence sweep and credit-spread sensitivity
(CS01) tables used in the project writeup. Run this directly:

    python analysis.py

Outputs:
    - divergence_sweep.csv   (TF vs single-rate price across moneyness)
    - cs01_sensitivity.csv   (CS01 by moneyness bucket, both treatments)
    - printed summary tables to stdout
"""

import csv
import numpy as np
from cb_pricer import CBTerms, price_single_rate, price_tf, fd_delta_tf, fd_delta_single

# ----------------------------------------------------------------------
# Base case: 5yr vanilla convertible, semi-annual coupon, conv. price $50
# ----------------------------------------------------------------------
BASE = CBTerms(
    face=100.0,
    coupon_rate=0.04,
    coupon_freq=2,
    maturity=5.0,
    conversion_ratio=2.0,      # conversion price = face / conversion_ratio = $50
    r_riskfree=0.045,
    credit_spread=0.03,        # 300 bp flat issuer spread
    vol=0.35,
    n_steps=300,
)


def run_divergence_sweep(terms=BASE, stock_prices=None, out_csv="divergence_sweep.csv"):
    if stock_prices is None:
        stock_prices = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 150, 200, 300]

    rows = []
    print(f"{'S0':>6} {'ConvVal':>9} {'Single':>9} {'TF':>9} {'Div %':>7}")
    for S0 in stock_prices:
        single_price = price_single_rate(S0, terms)
        tf_price, cocb, cotv = price_tf(S0, terms)
        div_pct = (tf_price - single_price) / single_price * 100
        conv_val = terms.conversion_ratio * S0
        rows.append({
            "S0": S0, "conversion_value": conv_val,
            "single_rate_price": single_price, "tf_price": tf_price,
            "divergence_pct": div_pct,
        })
        print(f"{S0:6.0f} {conv_val:9.2f} {single_price:9.3f} {tf_price:9.3f} {div_pct:7.2f}")

    peak = max(rows, key=lambda r: r["divergence_pct"])
    print(f"\nPeak divergence: S0={peak['S0']}, {peak['divergence_pct']:.2f}%")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return rows


def run_cs01_sensitivity(terms=BASE, out_csv="cs01_sensitivity.csv"):
    buckets = {
        "Deep OTM": 20, "Slightly OTM": 40, "Near boundary": 50,
        "Slightly ITM": 60, "Moderately ITM (peak div.)": 75, "Deep ITM": 200,
    }
    bumped = CBTerms(**{**terms.__dict__, "credit_spread": terms.credit_spread + 0.01})

    rows = []
    print(f"\n{'Region':>28} {'S0':>5} {'Delta_S':>8} {'Delta_TF':>9} {'CS01_S(%)':>11} {'CS01_TF(%)':>11}")
    for name, S0 in buckets.items():
        d_s = fd_delta_single(S0, terms)
        d_tf = fd_delta_tf(S0, terms)

        p_s0 = price_single_rate(S0, terms)
        p_s1 = price_single_rate(S0, bumped)
        cs01_s = (p_s1 - p_s0) / p_s0 * 100

        p_tf0, _, _ = price_tf(S0, terms)
        p_tf1, _, _ = price_tf(S0, bumped)
        cs01_tf = (p_tf1 - p_tf0) / p_tf0 * 100

        rows.append({
            "region": name, "S0": S0, "delta_single": d_s, "delta_tf": d_tf,
            "cs01_single_pct": cs01_s, "cs01_tf_pct": cs01_tf,
        })
        print(f"{name:>28} {S0:5.0f} {d_s:8.3f} {d_tf:9.3f} {cs01_s:11.3f} {cs01_tf:11.3f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    print("=== Divergence sweep: TF vs single-discount-rate, by moneyness ===\n")
    run_divergence_sweep()

    print("\n=== Credit-spread sensitivity (CS01, 100bp bump) by moneyness ===")
    run_cs01_sensitivity()
