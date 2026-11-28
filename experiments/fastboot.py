"""Fast paired two-level bootstrap (clusters x seeds) for AUROC differences.
Scores are binned into 4096 quantile bins per (seed, model, class); a bootstrap replicate is then a weighted sum of
per-cluster histograms, and the AUROC is computed from the two class histograms with ties inside a bin counted as 1/2.
Point estimates are exact (rank-based); only the interval uses the binned statistic."""
import numpy as np
NBIN = 4096
def _auc_exact(s, y):
    o = np.argsort(s, kind="mergesort"); ss = s[o]; _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = (cs - cnt + cs + 1) / 2.0; rk = np.empty(len(s)); rk[o] = avg[inv]; n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def _hist(s, pos, cl, nc):
    e = np.unique(np.quantile(s, np.linspace(0, 1, NBIN + 1)[1:-1])); b = np.searchsorted(e, s, side="left"); nb = len(e) + 1
    Hp = np.zeros((nc, nb)); Hn = np.zeros((nc, nb)); np.add.at(Hp, (cl[pos], b[pos]), 1.0); np.add.at(Hn, (cl[~pos], b[~pos]), 1.0); return Hp, Hn
def _auc_w(W, Hp, Hn):
    P = W @ Hp; N = W @ Hn; cn = np.cumsum(N, 1) - N; num = (P * (cn + 0.5 * N)).sum(1); den = P.sum(1) * N.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"): return np.where(den > 0, num / den, np.nan)
def scores(kind):
    if kind == "macro": return [(lambda P, k=k: P[:, k].astype(np.float64), (lambda y, k=k: y == k)) for k in range(3)]
    if kind == "abn": return [(lambda P: 1.0 - P[:, 0].astype(np.float64), lambda y: y != 0)]
    k = "NSV".index(kind[-1]); return [(lambda P: P[:, k].astype(np.float64), lambda y: y == k)]
def boot(PA, PB, y, clusters, kind="macro", nb=2000, seed=20261127):
    rng = np.random.default_rng(seed); ss = sorted(set(PA) & set(PB)); u, cl = np.unique(clusters, return_inverse=True); nc = len(u); sc = scores(kind)
    W = rng.multinomial(nc, np.ones(nc) / nc, size=nb).astype(np.float64)
    ex = {m: [] for m in "AB"}; D = np.zeros((nb, len(ss)))
    for j, s in enumerate(ss):
        aw = {}
        for m, Pm in (("A", PA[s]), ("B", PB[s])):
            vals, reps = [], []
            for f, g in sc:
                sv = f(Pm); pos = g(y); vals.append(_auc_exact(sv, pos.astype(int))); Hp, Hn = _hist(sv, pos, cl, nc); reps.append(_auc_w(W, Hp, Hn))
            ex[m].append(np.nanmean(vals))
            with np.errstate(invalid="ignore"): aw[m] = np.nanmean(np.stack(reps, 0), 0)
        D[:, j] = aw["A"] - aw["B"]
    pick = rng.integers(0, len(ss), size=(nb, len(ss))); d = np.take_along_axis(D, pick, 1).mean(1); d = d[~np.isnan(d)]
    a = float(np.mean(ex["A"])); b = float(np.mean(ex["B"]))
    return dict(a=a, b=b, diff=a - b, ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])], ci90=[float(v) for v in np.percentile(d, [5, 95])],
                p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), seeds=len(ss), records=int(nc), windows=int(len(y)))
