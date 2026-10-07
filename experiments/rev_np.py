"""NumPy-only post-hoc tools: decoder re-implementation, quantisers, channel
models, classical baselines (PCA, spectral summary, RR-interval features)."""
from __future__ import annotations
import numpy as np
from scipy.special import erf
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ------------------------------------------------------------------ decoder
def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def np_decoder(w, z):
    """Softmax class probabilities from the exported decoder weights (eval mode)."""
    h = gelu(z @ w["W1"].T + w["b1"]); h = gelu(h @ w["W2"].T + w["b2"])
    logits = h @ w["W3"].T + w["b3"]
    logits = logits - logits.max(1, keepdims=True); e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def load_run(path):
    d = np.load(path, allow_pickle=True)
    w = {k: d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")} if "W1" in d else None
    return d, w


# ------------------------------------------------------------------ quantisers
def quantize(z, bits, scale=None):
    """Uniform symmetric mid-tread quantiser on [-1, 1] (after optional per-dim scaling).
    Returns the de-quantised value (what the receiver reconstructs)."""
    qmax = float(max(1, 2 ** (bits - 1) - 1))
    zs = z / scale if scale is not None else z
    q = np.clip(np.round(zs * qmax), -qmax - 1, qmax) / qmax
    return q * scale if scale is not None else q


def quantize_int(z, bits, scale=None):
    """Integer codes (what is transmitted)."""
    qmax = int(max(1, 2 ** (bits - 1) - 1))
    zs = z / scale if scale is not None else z
    return np.clip(np.round(zs * qmax), -qmax - 1, qmax).astype(np.int32)


def dequantize_int(q, bits, scale=None):
    qmax = float(max(1, 2 ** (bits - 1) - 1))
    z = q.astype(np.float32) / qmax
    return z * scale if scale is not None else z


def dim_scale(Ztr, pct=99.9):
    """Per-dimension scale from training latents (robust max-abs)."""
    return np.maximum(np.percentile(np.abs(Ztr), pct, axis=0), 1e-6).astype(np.float32)


# ------------------------------------------------------------------ channels
def bsc_flip(q, bits, p, rng):
    """Binary symmetric channel on the two's-complement integer codes."""
    qmax = 2 ** (bits - 1) - 1
    u = (q + (qmax + 1)).astype(np.int64)                 # to unsigned [0, 2^bits)
    flips = (rng.random(u.shape + (bits,)) < p).astype(np.int64)
    mask = (flips << np.arange(bits)).sum(-1)
    return (u ^ mask) - (qmax + 1)


def bursty_bsc(q, bits, ber, rng, p_bad=0.1, p_bg=0.2):
    """Bursty bit flips: a Gilbert-Elliott chain over the coefficient sequence of
    each window; in the bad state every bit of a coefficient flips with prob p_bad,
    in the good state none. The chain is tuned so the mean BER equals `ber`."""
    if ber <= 0:
        return q.copy()
    pi_b = min(ber / p_bad, 0.999); p_gb = min(p_bg * pi_b / (1 - pi_b), 1.0)
    bad = gilbert_elliott_mask(q.shape, p_gb, p_bg, 0.0, 1.0, rng)
    qmax = 2 ** (bits - 1) - 1
    u = (q + (qmax + 1)).astype(np.int64)
    flips = (rng.random(u.shape + (bits,)) < np.where(bad, p_bad, 0.0)[..., None]).astype(np.int64)
    mask = (flips << np.arange(bits)).sum(-1)
    return (u ^ mask) - (qmax + 1)


def gilbert_elliott_mask(shape, p_gb, p_bg, e_good, e_bad, rng):
    """Correlated (bursty) error indicator over the LAST axis using a two-state
    Gilbert-Elliott chain: P(good->bad)=p_gb, P(bad->good)=p_bg; error prob e_good/e_bad."""
    n = shape[-1]; lead = int(np.prod(shape[:-1]))
    state = (rng.random(lead) < p_gb / (p_gb + p_bg)).astype(np.int8)  # stationary start
    out = np.zeros((lead, n), bool)
    for t in range(n):
        e = np.where(state == 1, e_bad, e_good)
        out[:, t] = rng.random(lead) < e
        flip = rng.random(lead) < np.where(state == 1, p_bg, p_gb)
        state = np.where(flip, 1 - state, state)
    return out.reshape(shape)


def fragment_erasure(zq, n_frag, per, rng, bursty=None):
    """Split the D coefficients into n_frag equal fragments (packets); erase whole
    fragments with probability `per` (iid) or with a Gilbert-Elliott process across
    fragments (bursty=(p_gb, p_bg, e_good, e_bad)). Erased coefficients -> 0."""
    N, D = zq.shape; f = D // n_frag
    if bursty is None:
        lost = rng.random((N, n_frag)) < per
    else:
        lost = gilbert_elliott_mask((N, n_frag), *bursty, rng)
    mask = np.repeat(~lost, f, axis=1)
    return zq * mask, lost.mean()


def rayleigh_per(snr_db_mean, snr_th_db, n_frag_per_decision=1):
    """Packet-erasure probability under block Rayleigh fading with an SNR threshold:
    P(SNR < th) = 1 - exp(-th/mean)."""
    return 1.0 - np.exp(-(10 ** (snr_th_db / 10)) / (10 ** (snr_db_mean / 10)))


# ------------------------------------------------------------------ readouts
def logreg_readout(Ftr, ytr, Fte, K=3, seed=0):
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Ftr), ytr)
    p = clf.predict_proba(sc.transform(Fte)); full = np.zeros((len(Fte), K))
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = p[:, j]
    return full


# ------------------------------------------------------------------ classical codes
def summary_features(X):
    """Original 11-per-channel summary (mean, std, rms + 8 low-frequency FFT bins)."""
    N, C, T = X.shape; feats = []
    for c in range(C):
        xc = X[:, c, :]; mag = np.abs(np.fft.rfft(xc, axis=1))[:, 1:9]
        feats.append(np.stack([xc.mean(1), xc.std(1), np.sqrt((xc ** 2).mean(1))], 1)); feats.append(mag)
    return np.concatenate(feats, 1)


def detect_r_peaks(x, fs):
    """Compact Pan-Tompkins-style detector (band-pass, derivative, squaring,
    moving-window integration, adaptive threshold, refractory period)."""
    b, a = butter(2, [5 / (fs / 2), min(15, fs / 2 - 1) / (fs / 2)], btype="band")
    y = filtfilt(b, a, x)
    d = np.diff(y, prepend=y[0]); s = d ** 2
    w = max(int(0.15 * fs), 1); m = np.convolve(s, np.ones(w) / w, mode="same")
    thr = 0.3 * np.max(m[int(0.5 * fs):]) if len(m) > fs else 0.3 * m.max()
    refr = int(0.25 * fs); peaks = []; i = 0
    while i < len(m):
        if m[i] > thr:
            j = i + np.argmax(m[i:i + refr]); peaks.append(j); i = j + refr
        else:
            i += 1
    return np.asarray(peaks)


def rr_features(X, fs):
    """22-d clinically motivated code per window from lead 0: RR statistics
    (mean, SD, RMSSD, pNN50, min, max, range, CV, irregularity), R-amplitude
    stats, QRS-energy proxy, plus 8 low-frequency spectral bins of the RR tachogram."""
    N = X.shape[0]; F = np.zeros((N, 22), np.float32)
    for i in range(N):
        x = X[i, 0]; p = detect_r_peaks(x, fs)
        if len(p) < 3:
            continue
        rr = np.diff(p) / fs; drr = np.diff(rr)
        amp = x[p]
        f = [rr.mean(), rr.std(), np.sqrt((drr ** 2).mean()) if len(drr) else 0, (np.abs(drr) > 0.05).mean() if len(drr) else 0,
             rr.min(), rr.max(), rr.max() - rr.min(), rr.std() / max(rr.mean(), 1e-6),
             np.median(np.abs(rr - np.median(rr))), len(p) / (len(x) / fs),
             amp.mean(), amp.std(), amp.min(), amp.max()]
        tach = np.interp(np.linspace(0, len(rr) - 1, 32), np.arange(len(rr)), rr) if len(rr) > 1 else np.zeros(32)
        spec = np.abs(np.fft.rfft(tach - tach.mean()))[1:9]
        F[i] = np.asarray(f + list(spec), np.float32)
    return F


def pca_code(Xtr, Xte, k, seed=0):
    Ftr = Xtr.reshape(len(Xtr), -1); Fte = Xte.reshape(len(Xte), -1)
    pca = PCA(n_components=k, random_state=seed).fit(Ftr)
    return pca.transform(Ftr), pca.transform(Fte)
