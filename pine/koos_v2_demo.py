"""
KOOS v2 -- Python reference implementation of the Pine v6 strategy.

Bridges:
  [1] Derbazi 2026a  Thm 3.5  -- kernel monotone / concave  =>  >=_lr, >=_lc
  [2] Derbazi 2026b  Prop 2.9 -- superlevel-set endpoint     =>  >=_st, >=_hr
  [3] Gnedin  2026   Thm 1    -- accept iff x_now <= 2/(T-t)  (beta-2 rule)
                     Thm 3    -- discrete cutoffs delta_k    (optional)

Demo: synthetic price series with a known regime shift; the gates should
fire only after the shift, and exits should obey Gnedin's hyperbolic boundary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1.  synthetic OHLC with a regime shift -----------------------
rng = np.random.default_rng(7)
N = 1200
mu_pre,  sd_pre  = 0.0000, 0.010      # flat-ish baseline
mu_post, sd_post = 0.0040, 0.010      # +0.4 sigma drift after bar 600
r = np.concatenate([rng.normal(mu_pre,  sd_pre,  600),
                    rng.normal(mu_post, sd_post, N - 600)])
price = 100 * np.exp(np.cumsum(r))

# ---------- 2.  strategy parameters --------------------------------------
nBase, nCurr = 200, 50
nBins        = 21
binHalf      = 2.5
Tbars        = 40
endpMrg      = 0.05

# Gnedin discrete cutoffs (Paper 3, Thm 3)
DELTA = np.array([1.353, 0.803, 0.572, 0.445, 0.363, 0.303,
                  0.266, 0.235, 0.210, 0.190, 0.173])

# ---------- 3.  per-bar gate evaluation ----------------------------------
def gates(window_curr, window_base):
    """Detect "current >=_X baseline" gates.
       Per Paper 2 Def 1.3 and Prop 2.9, set  P = baseline, Q = current;
       then  baseline <=_st current  <=>  ell(x0) >= 1  with {ell>=1} initial,
       where  ell = f_baseline / f_current.
       Returns (stDom, hrDom, lcShape, ell_at_x0, monoDownLR, concave_logL).
    """
    sig = np.std(np.concatenate([window_curr, window_base]))
    mu  = np.mean(np.concatenate([window_curr, window_base]))
    if sig <= 0:
        return False, False, False, np.nan, False, False
    edges = np.linspace(mu - binHalf*sig, mu + binHalf*sig, nBins + 1)
    fC, _ = np.histogram(window_curr, bins=edges)   # current  (= Q)
    fB, _ = np.histogram(window_base, bins=edges)   # baseline (= P)
    fC = (fC + 1) / (len(window_curr) + nBins)
    fB = (fB + 1) / (len(window_base) + nBins)
    # 3-bin moving-average smoothing (noise mitigation; preserves shape gates)
    kern = np.array([1, 2, 1]) / 4
    fC = np.convolve(fC, kern, mode="same"); fC /= fC.sum()
    fB = np.convolve(fB, kern, mode="same"); fB /= fB.sum()
    ell  = fB / fC                                  # ell = fP/fQ per Prop 2.9
    logL = np.log(ell)

    # Paper 1 Thm 3.5(i): baseline <=_lr current  <=>  ell nonincreasing in x
    monoDownLR = np.all(np.diff(logL) <= 0)
    # log-concavity of fP/fQ (Paper 1 Thm 3.5(ii) shape diagnostic)
    concave    = np.all(np.diff(logL, 2) <= 0)

    # Paper 2 Prop 2.9 -- leftmost non-trivial-P bin = x_0
    left = int(np.argmax(fB > 1.5 / (len(window_base) + nBins)))
    ellL = ell[left]
    endpointOK = ellL >= 1.0 + endpMrg

    # Paper 2 Prop 2.8 sign-pattern relaxation: ell-1 may have <=2 sign changes
    # and must be <0 on the rightmost run.  More forgiving than strict Prop 2.9.
    tol = 0.02
    signs = np.sign(ell[left:] - 1.0) * (np.abs(ell[left:] - 1.0) > tol)
    nz = signs[signs != 0]
    changes = int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0
    rightNeg = (nz[-1] < 0) if len(nz) > 0 else True
    intervalOK = (changes <= 2) and rightNeg

    # Prop 2.9 hr addition: ell nonincreasing on J\A (right tail)
    hrTail, seenOut = True, False
    for i in range(left, len(ell) - 1):
        if ell[i] < 1.0:
            seenOut = True
        if seenOut and ell[i+1] > ell[i] + 1e-9:
            hrTail = False
            break

    # debug stash
    gates._last = dict(endpointOK=endpointOK, intervalOK=intervalOK, hrTail=hrTail,
                       changes=changes, rightNeg=rightNeg, ellL=ellL)
    # ---- definition-level gates (gold standard) ----
    # current >=_st baseline  <=>  Sbar_curr(x) >= Sbar_base(x)  for all x
    n_curr_w = len(window_curr); n_base_w = len(window_base)
    # use a shared evaluation grid (midpoints of bins)
    Sbar_curr = np.array([(window_curr >= e).mean() for e in edges[:-1]])
    Sbar_base = np.array([(window_base >= e).mean() for e in edges[:-1]])
    stDom_def = np.all(Sbar_curr + 1e-9 >= Sbar_base)
    # current >=_hr baseline  <=>  Sbar_curr / Sbar_base nondecreasing
    eps = 1e-6
    ratio = (Sbar_curr + eps) / (Sbar_base + eps)
    hrDom_def = np.all(np.diff(ratio) >= -1e-9)

    # Paper 2 Prop 2.9 sufficient route (diagnostic; can fail even when defn holds)
    stDom_p29 = endpointOK and intervalOK
    hrDom_p29 = stDom_p29 and hrTail

    # debug stash
    gates._last = dict(endpointOK=endpointOK, intervalOK=intervalOK, hrTail=hrTail,
                       changes=changes, rightNeg=rightNeg, ellL=ellL,
                       stDom_def=stDom_def, hrDom_def=hrDom_def,
                       stDom_p29=stDom_p29, hrDom_p29=hrDom_p29)
    # entry uses the gold-standard >=_hr (definition); Prop 2.9 shown alongside
    stDom   = stDom_def
    hrDom   = hrDom_def
    lcShape = concave
    return stDom, hrDom, lcShape, ellL, monoDownLR, concave


# ---------- 4.  event-driven backtest ------------------------------------
pos    = 0            # 0 = flat, 1 = long
entry  = np.nan
t      = 0            # bars in position
log_t  = []           # diagnostics
trades = []           # (entry_bar, exit_bar, reason, pnl)

for i in range(nBase + nCurr, N):
    cur  = r[i - nCurr : i]
    base = r[i - nCurr - nBase : i - nCurr]
    stDom, hrDom, lrDom_lc, ellL, monoUp, concave = gates(cur, base)

    if pos == 1:
        t += 1
        x_now    = np.log(entry / price[i])           # loss if we exit now
        remain   = max(1, Tbars - t)
        thr      = 2.0 / remain                       # Gnedin (9)
        gnedin   = x_now <= thr
        timeexit = t >= Tbars
        flip     = not stDom
        if gnedin or timeexit or flip:
            reason = "beta2" if gnedin else ("T" if timeexit else "flip")
            trades.append((entry_bar, i, reason, price[i]/entry - 1))
            pos, t = 0, 0
    else:
        # entry gate: stochastic dominance of current over baseline
        if stDom:
            pos, entry, entry_bar, t = 1, price[i], i, 0

    L = gates._last
    log_t.append(dict(bar=i, price=price[i], stDom=stDom, hrDom=hrDom,
                      lrDom_lc=lrDom_lc, ell_x0=ellL, pos=pos,
                      stDom_p29=L["stDom_p29"], hrDom_p29=L["hrDom_p29"],
                      endpointOK=L["endpointOK"], intervalOK=L["intervalOK"]))

df = pd.DataFrame(log_t)
print("Gate firing rates (whole test region):")
print(df[["stDom","hrDom","stDom_p29","hrDom_p29","endpointOK","intervalOK"]]
        .mean().round(3).to_string())
print("\nDetection window (650<=bar<=850, baseline pre-shift / current post-shift):")
sub = df[(df.bar >= 650) & (df.bar <= 850)]
print(sub[["stDom","hrDom","stDom_p29","hrDom_p29"]].mean().round(3).to_string())
print(f"ell(x0) deciles:\n{df['ell_x0'].describe()}")
print(f"Trades: {len(trades)}")
wins = [p for _,_,_,p in trades if p > 0]
print(f"Win rate: {len(wins)}/{len(trades)} = {len(wins)/max(1,len(trades)):.0%}")
print(f"Total return: {sum(p for _,_,_,p in trades):+.2%}")
print(f"Mean PnL/trade: {np.mean([p for _,_,_,p in trades]):+.3%}" if trades else "")
print("\nReason breakdown:")
print(pd.Series([r for _,_,r,_ in trades]).value_counts().to_string())

# ---------- 5.  visualisation -------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 1, 1]))

ax = axes[0]
ax.plot(price, lw=0.8, color="black", label="price")
ax.axvline(600, color="grey", ls="--", lw=0.7, label="regime shift")
for eb, xb, reason, pnl in trades:
    c = "tab:green" if pnl > 0 else "tab:red"
    ax.axvspan(eb, xb, alpha=0.15, color=c)
    ax.scatter(eb, price[eb], marker="^", color="teal", s=40, zorder=5)
    ax.scatter(xb, price[xb], marker="x", color=c, s=40, zorder=5)
ax.set_title("KOOS v2  --  entries (▲), exits (✕), shaded by trade outcome")
ax.legend(loc="upper left"); ax.set_ylabel("price")

ax = axes[1]
ax.plot(df["bar"], df["ell_x0"], lw=0.7, color="purple")
ax.axhline(1.0 + endpMrg, color="black", ls=":", lw=0.6)
ax.axvline(600, color="grey", ls="--", lw=0.7)
ax.set_ylabel("ell(x₀)"); ax.set_title("Paper 2 Prop 2.9  endpoint  ℓ(x₀)")

ax = axes[2]
ax.fill_between(df["bar"], 0, df["stDom"].astype(int), step="post",
                color="teal", alpha=0.6, label=">=_st  (definition, entry gate)")
ax.fill_between(df["bar"], 0, df["stDom_p29"].astype(int), step="post",
                color="purple", alpha=0.5, label=">=_st  (Prop 2.9 sufficient)")
ax.fill_between(df["bar"], 0, df["endpointOK"].astype(int) * 0.5, step="post",
                color="orange", alpha=0.4, label="ell(x0) >= 1+eps (endpoint only)")
ax.axvline(600, color="grey", ls="--", lw=0.7)
ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("bar"); ax.legend(loc="upper left", fontsize=8)
ax.set_title("Stochastic-order gates active (1) / inactive (0)")

plt.tight_layout()
plt.savefig("/tmp/koos_v2_demo.png", dpi=130)
print("\nSaved /tmp/koos_v2_demo.png")
