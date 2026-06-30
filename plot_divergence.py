"""
Generates divergence_chart.png: TF vs single-rate price, and the divergence
percentage, across moneyness. Run with: python plot_divergence.py
"""

import numpy as np
import matplotlib.pyplot as plt
from cb_pricer import CBTerms, price_single_rate, price_tf

TERMS = CBTerms(
    face=100.0, coupon_rate=0.04, coupon_freq=2, maturity=5.0,
    conversion_ratio=2.0, r_riskfree=0.045, credit_spread=0.03,
    vol=0.35, n_steps=300,
)

stock_prices = np.arange(5, 205, 5)
single_prices = [price_single_rate(S0, TERMS) for S0 in stock_prices]
tf_prices = [price_tf(S0, TERMS)[0] for S0 in stock_prices]
divergence_pct = [(tf - s) / s * 100 for tf, s in zip(tf_prices, single_prices)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.plot(stock_prices, single_prices, label="Single risky discount rate", linewidth=2)
ax1.plot(stock_prices, tf_prices, label="Tsiveriotis-Fernandes", linewidth=2)
ax1.set_ylabel("Convertible bond price")
ax1.set_title("Convertible Bond Price: Single-Rate vs. Tsiveriotis-Fernandes")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(stock_prices, divergence_pct, color="darkred", linewidth=2)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlabel("Stock price (S0)")
ax2.set_ylabel("TF - Single-rate divergence (%)")
ax2.set_title("Pricing Divergence by Moneyness")
ax2.grid(alpha=0.3)

peak_idx = int(np.argmax(divergence_pct))
ax2.annotate(
    f"Peak: {divergence_pct[peak_idx]:.2f}% at S0={stock_prices[peak_idx]}",
    xy=(stock_prices[peak_idx], divergence_pct[peak_idx]),
    xytext=(stock_prices[peak_idx] + 20, divergence_pct[peak_idx] - 1.5),
    arrowprops=dict(arrowstyle="->", color="gray"),
)

plt.tight_layout()
plt.savefig("divergence_chart.png", dpi=150)
print("Saved divergence_chart.png")
