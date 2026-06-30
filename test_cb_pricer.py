"""
Sanity checks for cb_pricer.py.

These aren't a full unit-test suite — they check that the lattice converges
to the limiting cases you'd expect from the theory, which is the minimum bar
for trusting the divergence/CS01 numbers in analysis.py.

Run with: python test_cb_pricer.py
"""

import numpy as np
from cb_pricer import CBTerms, price_single_rate, price_tf, fd_delta_tf, fd_delta_single

TERMS = CBTerms(
    face=100.0, coupon_rate=0.04, coupon_freq=2, maturity=5.0,
    conversion_ratio=2.0, r_riskfree=0.045, credit_spread=0.03,
    vol=0.35, n_steps=300,
)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def test_deep_itm_delta_converges_to_conversion_ratio():
    S0 = 1000  # deep, deep ITM
    d_tf = fd_delta_tf(S0, TERMS)
    d_single = fd_delta_single(S0, TERMS)
    ok = abs(d_tf - TERMS.conversion_ratio) < 0.02 and abs(d_single - TERMS.conversion_ratio) < 0.02
    return check(
        "Deep ITM delta -> conversion ratio",
        ok,
        f"delta_tf={d_tf:.4f}, delta_single={d_single:.4f}, conversion_ratio={TERMS.conversion_ratio}",
    )


def test_deep_otm_price_converges_to_straight_bond_floor():
    """Deep OTM, the convertible should price close to a straight (non-convertible)
    bond discounted at the risky rate, since conversion is essentially worthless."""
    S0 = 0.01
    price = price_single_rate(S0, TERMS)

    # discount straight bond cash flows at the risky rate as an approximate floor
    n_coupons = int(round(TERMS.maturity * TERMS.coupon_freq))
    cpn = TERMS.face * TERMS.coupon_rate / TERMS.coupon_freq
    risky_rate = TERMS.r_riskfree + TERMS.credit_spread
    floor = sum(cpn * np.exp(-risky_rate * (k / TERMS.coupon_freq)) for k in range(1, n_coupons + 1))
    floor += TERMS.face * np.exp(-risky_rate * TERMS.maturity)

    ok = abs(price - floor) / floor < 0.01
    return check(
        "Deep OTM price -> straight bond floor",
        ok,
        f"lattice price={price:.3f}, analytic floor={floor:.3f}",
    )


def test_tf_geq_single_rate_everywhere():
    """TF should never price below the single-rate model: TF gives partial credit
    relief on the equity-linked component, single-rate discounts everything at the
    risky rate, so TF >= single-rate always (for credit_spread >= 0)."""
    ok = True
    for S0 in [5, 20, 50, 100, 300]:
        single = price_single_rate(S0, TERMS)
        tf, _, _ = price_tf(S0, TERMS)
        if tf < single - 1e-6:
            ok = False
    return check("TF price >= single-rate price at all moneyness levels tested", ok)


def test_zero_credit_spread_collapses_treatments():
    """With credit_spread = 0, both treatments discount everything at the risk-free
    rate, so single-rate and TF should converge to (nearly) the same price."""
    zero_spread = CBTerms(**{**TERMS.__dict__, "credit_spread": 0.0})
    S0 = 50
    single = price_single_rate(S0, zero_spread)
    tf, _, _ = price_tf(S0, zero_spread)
    ok = abs(tf - single) / single < 0.005
    return check(
        "Zero credit spread collapses TF and single-rate to ~same price",
        ok,
        f"single={single:.4f}, tf={tf:.4f}",
    )


def test_lattice_converges_with_more_steps():
    """Price shouldn't change much between 150 and 300 steps if the lattice has converged."""
    coarse = CBTerms(**{**TERMS.__dict__, "n_steps": 150})
    fine = CBTerms(**{**TERMS.__dict__, "n_steps": 300})
    S0 = 50
    p_coarse = price_single_rate(S0, coarse)
    p_fine = price_single_rate(S0, fine)
    ok = abs(p_fine - p_coarse) / p_fine < 0.01
    return check(
        "Price stable under step-count refinement (150 -> 300 steps)",
        ok,
        f"150 steps={p_coarse:.4f}, 300 steps={p_fine:.4f}",
    )


if __name__ == "__main__":
    results = [
        test_deep_itm_delta_converges_to_conversion_ratio(),
        test_deep_otm_price_converges_to_straight_bond_floor(),
        test_tf_geq_single_rate_everywhere(),
        test_zero_credit_spread_collapses_treatments(),
        test_lattice_converges_with_more_steps(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
