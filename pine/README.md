# KOOS v2 — Kernel-Ordered Optimal Stopping

A TradingView Pine Script v6 strategy that bridges three recent papers:

1. **Derbazi (2026a)** *Kernel Characterisations of Stochastic Orders Within Parametric Density Families.* Theorem 3.5 (kernel monotone/concave → likelihood-ratio / relative-log-concavity orders).
2. **Derbazi (2026b)** *Stochastic Ordering under Weaker Likelihood-Ratio Shape Conditions.* Proposition 2.9 (superlevel-set endpoint criterion for $\ge_{\mathrm{st}}/\ge_{\mathrm{hr}}$).
3. **Gnedin (2026)** *Optimal Stopping for the Uniform Distribution.* Theorem 1 (planar-Poisson optimal-stopping rule with hyperbolic boundary $2/(T-t)$) and Theorem 3 (discrete cutoffs $\delta_k$).

## Files

| File | Purpose |
|---|---|
| `koos_v2.pine` | Pine Script v6 strategy (deployable on TradingView) |
| `koos_v2_demo.py` | Python reference implementation + synthetic regime-shift backtest |
| `koos_ab_compare.py` | A/B/C/D comparison of entry-gate variants |
| `koos_v2_demo.png` | Output chart from the Python demo |

## Strategy logic

**Entry.** Direct empirical $\ge_{\mathrm{st}}$ test: the current return distribution stochastically dominates the baseline window when the empirical survival function $\bar F_{\text{curr}}(x) \ge \bar F_{\text{base}}(x)$ at every grid point. Paper 2 Prop 2.9 and Paper 1 kernel monotonicity/concavity appear as **diagnostic overlays**, not gates — they are sufficient-but-not-necessary conditions and fire ~0% of the time on noisy empirical histograms (A/B comparison in `koos_ab_compare.py` confirms).

**Exit.** Gnedin's beta-2 rule from Theorem 1: define the "draw value" at the current bar as the adverse log-excursion $x_{\text{now}} = \log(p_{\text{entry}}/p_{\text{close}})$ since entry, and exit (accept) once $x_{\text{now}} \le 2/(T-t)$ where $t$ is bars-in-position and $T$ is the horizon. As $T - t \to 0$ the threshold blows up and the position is forcibly liquidated.

**Optional discrete mode.** Setting `useDelta = true` swaps in the discrete-uniform cutoffs $\delta_k$ from Theorem 3 of paper 3, used to discretise the adverse excursion into integer ranks $k = \lfloor x_{\text{now}}/\sigma \rfloor$.

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
