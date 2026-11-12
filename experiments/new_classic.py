"""Classical transform codecs (DWT db4 / DCT) at matched byte budgets, with an idealised entropy coder.
Per 10-s two-lead window: orthonormal transform of each lead, uniform scalar quantisation with one step
size for the window, and the bit cost charged as the zeroth-order empirical entropy of the quantised
symbols (N * H(q)) plus 2 bytes for the step size.  This is the cost an adaptive arithmetic coder
approaches and it ignores model/side costs, so it favours the classical codec.  The step size is found
by bisection so that the cost meets the byte budget."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, math, numpy as np, pywt
from scipy.fft import dct, idct
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BUDGETS = [16, 32, 64, 128, 256, 512, 1024]
def fwd(x, kind):
    if kind == "dct": return dct(x, norm="ortho"), None
    c = pywt.wavedec(x, "db4", mode="periodization", level=5); arr, sl = pywt.coeffs_to_array(c); return arr, sl
def inv(a, sl, kind, T):
    if kind == "dct": return idct(a, norm="ortho")
    return pywt.waverec(pywt.array_to_coeffs(a, sl, output_format="wavedec"), "db4", mode="periodization")[:T]
def cost_bytes(q):
    _, cnt = np.unique(q, return_counts=True); p = cnt / q.size
    return (-(cnt * np.log2(p)).sum()) / 8.0 + 2.0
def quant_to_budget(c, B):
    lo, hi = 1e-4, float(np.abs(c).max()) * 2 + 1e-6          # step: small -> many bytes
    for _ in range(28):
        mid = math.sqrt(lo * hi); q = np.round(c / mid)
        if cost_bytes(q) > B: lo = mid
        else: hi = mid
    q = np.round(c / hi); return q * hi, cost_bytes(q), int((q != 0).sum())
def codec(X, kind, B):
    out = np.empty_like(X); T = X.shape[2]; nb = []; nz = []
    for i in range(X.shape[0]):
        cs = [fwd(X[i, c].astype(np.float64), kind) for c in range(X.shape[1])]
        allc = np.concatenate([a for a, _ in cs]); rec, b, k = quant_to_budget(allc, B); nb.append(b); nz.append(k)
        n = len(cs[0][0])
        for c in range(X.shape[1]):
            out[i, c] = inv(rec[c * n:(c + 1) * n], cs[c][1], kind, T)
    return out, float(np.mean(nb)), float(np.mean(nz))
def prd(X, Xr): return float(100 * np.sqrt(((X - Xr) ** 2).sum() / (X ** 2).sum()))
if __name__ == "__main__":
    d = np.load("caches/mitbih_fs125_both.npz", allow_pickle=True); X = d["X"]
    from rev_splits import DS2
    te = np.isin(d["groups"], DS2); res = json.load(open((WORK + "/classic_prd.json"))) if os.path.exists((WORK + "/classic_prd.json")) else {}
    if len(sys.argv) > 1: BUDGETS = [int(v) for v in sys.argv[1:]]
    for kind in ("dwt", "dct"):
        for B in BUDGETS:
            p = "caches/cls_%s_%d.npz" % (kind, B)
            if "%s_%d" % (kind, B) in res and os.path.exists(p): continue
            Xr, mb, mk = codec(X, kind, B); Xr = Xr.astype(np.float32)
            np.savez_compressed(p, X=Xr, y=d["y"], groups=d["groups"])
            res["%s_%d" % (kind, B)] = dict(budget=B, mean_bytes=mb, mean_nonzero=mk, prd_test=prd(X[te], Xr[te]), cr=5000 / B)
            print(kind, B, res["%s_%d" % (kind, B)], flush=True)
            json.dump(res, open((WORK + "/classic_prd.json"), "w"), indent=1)
