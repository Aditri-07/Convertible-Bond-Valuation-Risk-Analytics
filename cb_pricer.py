"""
Convertible bond pricer using a recombining binomial lattice.

Implements two credit treatments:
  1. Single risky discount rate (entire bond value discounted at r + credit spread)
  2. Tsiveriotis-Fernandes (1998): splits value into a "cash-only" (COCB) component
     discounted at the risky rate, and an equity-linked (COTV) component discounted
     at the risk-free rate. State-dependent: which component dominates depends on
     whether the bond is more likely to be converted or redeemed at each node.

Vanilla convertible: American-style conversion right, no call/put features.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CBTerms:
    face: float = 100.0
    coupon_rate: float = 0.04       # annual coupon rate
    coupon_freq: int = 2            # payments per year
    maturity: float = 5.0           # years
    conversion_ratio: float = 1.0   # shares per bond (we work in price-per-100 terms,
                                     # so conversion_ratio is effectively conversion_price-implied)
    r_riskfree: float = 0.045       # risk-free rate
    credit_spread: float = 0.03     # issuer credit spread (flat)
    vol: float = 0.35               # equity volatility
    n_steps: int = 200              # lattice steps


def build_stock_tree(S0, vol, r, T, n_steps):
    dt = T / n_steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    return dt, u, d, p


def coupon_at_step(step, dt, terms: CBTerms):
    """Return coupon cash paid at this step (approx: paid at lattice dates nearest
    actual coupon dates, evenly distributed)."""
    coupon_dt = 1.0 / terms.coupon_freq
    coupon_amt = terms.face * terms.coupon_rate / terms.coupon_freq
    # pay a coupon if this step crosses a coupon date (approx via nearest-step mapping)
    n_coupons = int(round(terms.maturity * terms.coupon_freq))
    coupon_steps = set()
    for k in range(1, n_coupons + 1):
        cdate = k * coupon_dt
        coupon_steps.add(int(round(cdate / dt)))
    return coupon_amt if step in coupon_steps else 0.0


def price_single_rate(S0, terms: CBTerms):
    dt, u, d, p = build_stock_tree(S0, terms.vol, terms.r_riskfree, terms.maturity, terms.n_steps)
    n = terms.n_steps
    disc_risky = np.exp(-(terms.r_riskfree + terms.credit_spread) * dt)

    # terminal stock prices
    S = S0 * d ** np.arange(n, -1, -1) * u ** np.arange(0, n + 1, 1)
    conv_val = terms.conversion_ratio * S
    redemption = terms.face + (terms.face * terms.coupon_rate / terms.coupon_freq if coupon_at_step(n, dt, terms) > 0 else 0.0)
    V = np.maximum(conv_val, redemption)

    for step in range(n - 1, -1, -1):
        S = S0 * d ** np.arange(step, -1, -1) * u ** np.arange(0, step + 1, 1)
        cont = disc_risky * (p * V[1:step + 2] + (1 - p) * V[0:step + 1])
        cpn = coupon_at_step(step, dt, terms)
        cont = cont + cpn
        conv_val = terms.conversion_ratio * S
        V = np.maximum(conv_val, cont)

    return V[0]


def price_tf(S0, terms: CBTerms):
    """Tsiveriotis-Fernandes split-component lattice."""
    dt, u, d, p = build_stock_tree(S0, terms.vol, terms.r_riskfree, terms.maturity, terms.n_steps)
    n = terms.n_steps
    disc_rf = np.exp(-terms.r_riskfree * dt)
    disc_risky = np.exp(-(terms.r_riskfree + terms.credit_spread) * dt)

    S = S0 * d ** np.arange(n, -1, -1) * u ** np.arange(0, n + 1, 1)
    conv_val = terms.conversion_ratio * S
    cpn_term = terms.face * terms.coupon_rate / terms.coupon_freq if coupon_at_step(n, dt, terms) > 0 else 0.0
    redemption = terms.face + cpn_term

    # at maturity: decide equity vs debt classification
    is_equity = conv_val > redemption
    COCB = np.where(is_equity, 0.0, redemption)
    COTV = np.where(is_equity, conv_val, 0.0)

    for step in range(n - 1, -1, -1):
        S = S0 * d ** np.arange(step, -1, -1) * u ** np.arange(0, step + 1, 1)
        cpn = coupon_at_step(step, dt, terms)

        cocb_cont = disc_risky * (p * COCB[1:step + 2] + (1 - p) * COCB[0:step + 1]) + cpn
        cotv_cont = disc_rf * (p * COTV[1:step + 2] + (1 - p) * COTV[0:step + 1])

        total = cocb_cont + cotv_cont
        conv_val = terms.conversion_ratio * S

        convert = conv_val > total
        COCB = np.where(convert, 0.0, cocb_cont)
        COTV = np.where(convert, conv_val, cotv_cont)

    total0 = COCB[0] + COTV[0]
    return total0, COCB[0], COTV[0]


def fd_delta_tf(S0, terms: CBTerms, bump=0.5):
    """Central finite-difference delta for the TF model."""
    up, _, _ = price_tf(S0 + bump, terms)
    down, _, _ = price_tf(S0 - bump, terms)
    return (up - down) / (2 * bump)


def fd_delta_single(S0, terms: CBTerms, bump=0.5):
    """Central finite-difference delta for the single-rate model."""
    up = price_single_rate(S0 + bump, terms)
    down = price_single_rate(S0 - bump, terms)
    return (up - down) / (2 * bump)
