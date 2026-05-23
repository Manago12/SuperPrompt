"""A/B/C comparison: entry-gate variant for KOOS."""
import numpy as np, pandas as pd

rng = np.random.default_rng(7)
N = 1200
r = np.concatenate([rng.normal(0.0000, 0.010, 600),
                    rng.normal(0.0040, 0.010, N - 600)])
price = 100 * np.exp(np.cumsum(r))

nBase, nCurr, nBins, binHalf, Tbars, endpMrg = 200, 50, 21, 2.5, 40, 0.05

def evaluate(window_curr, window_base):
    """Return dict of all three gate variants."""
    sig = np.std(np.concatenate([window_curr, window_base]))
    mu  = np.mean(np.concatenate([window_curr, window_base]))
    if sig <= 0:
        return dict(A=False, B=False, C=False)
    edges = np.linspace(mu - binHalf*sig, mu + binHalf*sig, nBins + 1)
    fC, _ = np.histogram(window_curr, bins=edges)
    fB, _ = np.histogram(window_base, bins=edges)
    fC = (fC + 1) / (len(window_curr) + nBins)
    fB = (fB + 1) / (len(window_base) + nBins)
    kern = np.array([1,2,1]) / 4
    fC = np.convolve(fC, kern, mode="same"); fC /= fC.sum()
    fB = np.convolve(fB, kern, mode="same"); fB /= fB.sum()
    ell = fB / fC

    # ---- A: Prop 2.9 (strict sufficient condition) ----
    left = int(np.argmax(fB > 1.5 / (len(window_base) + nBins)))
    endpointOK = ell[left] >= 1.0 + endpMrg
    inA_, intervalOK = True, True
    for v in ell[left:]:
        if inA_ and v < 1.0: inA_ = False
        elif (not inA_) and v >= 1.0: intervalOK = False; break
    A = endpointOK and intervalOK

    # ---- B: direct empirical >=_st  ----
    Sc = np.array([(window_curr >= e).mean() for e in edges[:-1]])
    Sb = np.array([(window_base >= e).mean() for e in edges[:-1]])
    B = bool(np.all(Sc + 1e-9 >= Sb))

    # ---- C: direct empirical >=_hr  ----
    eps = 1e-6
    ratio = (Sc + eps) / (Sb + eps)
    C = B and bool(np.all(np.diff(ratio) >= -1e-9))
    return dict(A=A, B=B, C=C)

def backtest(gate_name):
    pos, t, entry, entry_bar = 0, 0, np.nan, None
    trades, fired = [], 0
    for i in range(nBase + nCurr, N):
        cur  = r[i - nCurr : i]
        base = r[i - nCurr - nBase : i - nCurr]
        g = evaluate(cur, base)
        if g[gate_name]: fired += 1
        if pos == 1:
            t += 1
            x_now = np.log(entry / price[i])
            thr   = 2.0 / max(1, Tbars - t)
            if x_now <= thr or t >= Tbars:
                reason = "beta2" if x_now <= thr else "T"
                trades.append((entry_bar, i, reason, price[i]/entry - 1))
                pos, t = 0, 0
        else:
            if g[gate_name]:
                pos, entry, entry_bar, t = 1, price[i], i, 0
    pnl = [p for _,_,_,p in trades]
    return dict(gate=gate_name, fired_bars=fired, trades=len(trades),
                wins=sum(1 for p in pnl if p > 0),
                win_rate=sum(1 for p in pnl if p>0)/max(1,len(pnl)),
                total_ret=sum(pnl), mean_pnl=np.mean(pnl) if pnl else 0.0,
                worst=min(pnl) if pnl else 0.0, best=max(pnl) if pnl else 0.0)

def evaluate2(window_curr, window_base):
    g = evaluate(window_curr, window_base)
    # Hybrid D: direct >=_st  AND  Prop 2.9 endpoint half (ell(x0) >= 1+eps)
    sig = np.std(np.concatenate([window_curr, window_base]))
    mu  = np.mean(np.concatenate([window_curr, window_base]))
    edges = np.linspace(mu - binHalf*sig, mu + binHalf*sig, nBins + 1)
    fC, _ = np.histogram(window_curr, bins=edges)
    fB, _ = np.histogram(window_base, bins=edges)
    fC = (fC + 1)/(len(window_curr)+nBins); fB = (fB + 1)/(len(window_base)+nBins)
    kern = np.array([1,2,1])/4
    fC = np.convolve(fC,kern,mode="same"); fC/=fC.sum()
    fB = np.convolve(fB,kern,mode="same"); fB/=fB.sum()
    ell = fB/fC
    left = int(np.argmax(fB > 1.5/(len(window_base)+nBins)))
    endpointOK = ell[left] >= 1.0 + endpMrg
    g["D"] = g["B"] and endpointOK
    return g

def backtest2(gate_name):
    pos,t,entry,entry_bar = 0,0,np.nan,None
    trades, fired = [], 0
    for i in range(nBase + nCurr, N):
        cur, base = r[i-nCurr:i], r[i-nCurr-nBase:i-nCurr]
        g = evaluate2(cur, base)
        if g[gate_name]: fired += 1
        if pos == 1:
            t += 1
            x_now = np.log(entry/price[i])
            thr = 2.0/max(1, Tbars-t)
            if x_now <= thr or t >= Tbars:
                trades.append((entry_bar,i,"beta2" if x_now<=thr else "T",
                               price[i]/entry-1))
                pos, t = 0, 0
        else:
            if g[gate_name]:
                pos,entry,entry_bar,t = 1, price[i], i, 0
    pnl = [p for _,_,_,p in trades]
    return dict(gate=gate_name, fired_bars=fired, trades=len(trades),
                wins=sum(1 for p in pnl if p>0),
                win_rate=sum(1 for p in pnl if p>0)/max(1,len(pnl)),
                total_ret=sum(pnl), mean_pnl=np.mean(pnl) if pnl else 0.0,
                worst=min(pnl) if pnl else 0.0, best=max(pnl) if pnl else 0.0)

rows = [backtest2(g) for g in ("A","B","C","D")]
labels = {"A":"Prop 2.9   (endpoint + interval)",
          "B":"Direct  >=_st  (empirical survival)",
          "C":"Direct  >=_hr  (survival-ratio monotone)",
          "D":"Hybrid:  B  AND  ell(x0)>=1+eps"}
df = pd.DataFrame(rows)
df["gate_desc"] = df["gate"].map(labels)
df = df[["gate","gate_desc","fired_bars","trades","wins","win_rate",
         "total_ret","mean_pnl","worst","best"]]
print(df.to_string(index=False,
      formatters={"win_rate":"{:.0%}".format,
                  "total_ret":"{:+.2%}".format,
                  "mean_pnl":"{:+.3%}".format,
                  "worst":"{:+.2%}".format, "best":"{:+.2%}".format}))
