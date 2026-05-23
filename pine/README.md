# KOOS v2 — Kernel-Ordered Optimal Stopping (indicator)

A TradingView Pine Script v6 **indicator** (no order placement) that surfaces
diagnostics from three recent papers:

1. **Derbazi (2026a)** *Kernel Characterisations of Stochastic Orders Within Parametric Density Families.* Theorem 3.5 (kernel monotone/concave → likelihood-ratio / relative-log-concavity orders).
2. **Derbazi (2026b)** *Stochastic Ordering under Weaker Likelihood-Ratio Shape Conditions.* Proposition 2.9 (superlevel-set endpoint criterion for $\ge_{\mathrm{st}}/\ge_{\mathrm{hr}}$).
3. **Gnedin (2026)** *Optimal Stopping for the Uniform Distribution.* Theorem 1 (planar-Poisson optimal-stopping rule with hyperbolic boundary $2/(T-t)$) and Theorem 3 (discrete cutoffs $\delta_k$).

## Files

| File | Purpose |
|---|---|
| `koos_v2.pine` | Pine v6 indicator: paints background when $\ge_{\mathrm{st}}$ holds, surfaces $\ell(x_0)$, Prop 2.9 verdict, kernel monotone/concave flags, and a $2/(T-t)$ reference value |
| `koos_v2_demo.py` | Python reference for the same diagnostics + a synthetic regime-shift backtest (the backtest is illustrative; not part of the indicator) |
| `koos_ab_compare.py` | A/B/C/D comparison of entry-gate variants used to choose the indicator's headline signal |
| `koos_v2_demo.png` | Output chart from the Python demo |

## Indicator logic

**Headline signal.** Background tint when the current return distribution empirically $\ge_{\mathrm{st}}$ the baseline window: $\bar F_{\text{curr}}(x) \ge \bar F_{\text{base}}(x)$ at every grid point.

**Diagnostics shown alongside.**
- $\ell(x_0)$ — likelihood ratio at the leftmost non-trivial bin (Prop 2.9 endpoint).
- `endpointOK` — whether $\ell(x_0) \ge 1+\varepsilon$.
- `intervalOK` — whether $\{\ell \ge 1\}$ forms an initial interval of the support.
- `Prop 2.9 st` — the strict sufficient condition combining both above.
- `K monotone`, `K concave` — Paper 1 Thm 3.5 kernel-shape diagnostics (in this setting, the log-likelihood ratio plays the role of the kernel).
- `beta2 demo` — illustrative value of the Gnedin Theorem 1 threshold $2/(T-t)$ for the current bar-in-horizon counter.

No orders are placed.

## Why direct $\ge_{\mathrm{st}}$ and not Prop 2.9?

On a synthetic regime-shift test (1200 bars, +0.4σ drift after bar 600), the four entry-gate variants score as follows:

| Gate | Trades | Win rate | Total return |
|---|---:|---:|---:|
| Prop 2.9 (LR endpoint + interval shape) | 0 | — | +0.00% |
| **Direct $\ge_{\mathrm{st}}$ (empirical survival)** | **60** | **63%** | **+19.44%** |
| Direct $\ge_{\mathrm{hr}}$ (survival-ratio monotone) | 0 | — | +0.00% |
| Hybrid: $\ge_{\mathrm{st}}$ ∧ $\ell(x_0) \ge 1+\varepsilon$ | 2 | 0% | −1.04% |

Reproduce with `python3 pine/koos_ab_compare.py`.

The result is consistent with the papers, not a contradiction: Prop 2.9 is a *sufficient* condition for stochastic dominance, not a definition. Empirically, the gold standard is the direct survival-function comparison; the paper's theoretical contribution is to identify when a single endpoint check would have sufficed.
