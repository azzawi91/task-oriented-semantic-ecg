"""Minimal, dependency-free reader for MIT-BIH format-212 records and WFDB
annotation files. Covers exactly what the MIT-BIH Arrhythmia Database needs
(2-channel, format 212, .atr annotations with rhythm aux notes), so the
revision pipeline does not depend on the `wfdb` package.

Verified against wfdb-python on record 100 (2273 beats: 2239 N, 33 A, 1 V).
"""
from __future__ import annotations
import os
import numpy as np

# WFDB label_store -> symbol (subset that occurs in mitdb)
_SYM = {0: " ", 1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S",
        10: "E", 11: "j", 12: "/", 13: "Q", 14: "~", 16: "|", 18: "s", 19: "T", 20: "*",
        21: "D", 22: '"', 23: "=", 24: "p", 25: "B", 26: "^", 27: "t", 28: "+", 29: "u",
        30: "?", 31: "!", 32: "[", 33: "]", 34: "e", 35: "n", 36: "@", 37: "x", 38: "f",
        39: "(", 40: ")", 41: "r"}


def read_header(base: str):
    """Parse <base>.hea -> dict(fs, n_sig, n_samp, gains, baselines, names, fmt)."""
    with open(base + ".hea") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    rec = lines[0].split()
    n_sig, fs, n_samp = int(rec[1]), float(rec[2]), int(rec[3])
    gains, bases, names, fmts, files = [], [], [], [], []
    for l in lines[1:1 + n_sig]:
        p = l.split()
        files.append(p[0]); fmts.append(int(p[1]))
        g = p[2]
        if "(" in g:                      # gain(baseline)/units
            gain = float(g.split("(")[0]); base = int(g.split("(")[1].split(")")[0])
        else:
            gain = float(g.split("/")[0]); base = int(p[4])   # ADC zero as baseline
        gains.append(gain if gain != 0 else 200.0); bases.append(base)
        names.append(p[8] if len(p) > 8 else "ch%d" % len(names))
    return dict(fs=fs, n_sig=n_sig, n_samp=n_samp, gains=gains, baselines=bases,
                names=names, fmts=fmts, files=files)


def read_signal_212(base: str, hdr=None):
    """Read a 2-channel format-212 .dat -> physical signal [n_samp, 2] float32 (mV)."""
    hdr = hdr or read_header(base)
    assert all(f == 212 for f in hdr["fmts"]), "only format 212 supported"
    raw = np.fromfile(os.path.join(os.path.dirname(base), hdr["files"][0]), dtype=np.uint8)
    raw = raw[: (len(raw) // 3) * 3].reshape(-1, 3).astype(np.int32)
    s0 = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    s1 = raw[:, 2] | ((raw[:, 1] & 0xF0) << 4)
    s0 = np.where(s0 > 2047, s0 - 4096, s0)
    s1 = np.where(s1 > 2047, s1 - 4096, s1)
    n = hdr["n_samp"]
    if hdr["n_sig"] == 2:
        sig = np.stack([s0, s1], 1)[:n]
    else:
        sig = np.concatenate([s0, s1])[:n, None]
    phys = np.empty(sig.shape, np.float32)
    for c in range(sig.shape[1]):
        phys[:, c] = (sig[:, c] - hdr["baselines"][c]) / hdr["gains"][c]
    return phys


def read_signal_16(base: str, hdr=None):
    """Read an interleaved format-16 (int16 LE) multi-channel .dat -> [n_samp, n_sig] float32 (physical units)."""
    hdr = hdr or read_header(base)
    assert all(f == 16 for f in hdr["fmts"]), "only format 16 supported here"
    raw = np.fromfile(os.path.join(os.path.dirname(base), hdr["files"][0]), dtype="<i2")
    n, C = hdr["n_samp"], hdr["n_sig"]
    sig = raw[: n * C].reshape(n, C).astype(np.float32)
    for c in range(C):
        sig[:, c] = (sig[:, c] - hdr["baselines"][c]) / hdr["gains"][c]
    return sig


def read_signal(base: str):
    hdr = read_header(base)
    return (read_signal_212(base, hdr) if hdr["fmts"][0] == 212 else read_signal_16(base, hdr)), hdr


def read_annotation(base: str, ext: str = "atr"):
    """Read a WFDB annotation file -> (sample [N] int64, symbol [N] str, aux_note [N] str)."""
    b = np.fromfile(base + "." + ext, dtype=np.uint8)
    b = b[: (len(b) // 2) * 2].reshape(-1, 2)
    sample, symbol, aux = [], [], []
    total, i, n = 0, 0, b.shape[0]
    while i < n - 1:
        # core field(s), possibly preceded by SKIP words
        diff = 0
        while (int(b[i, 1]) >> 2) == 59:
            skip = (int(b[i + 1, 0]) << 16) + (int(b[i + 1, 1]) << 24) + int(b[i + 2, 0]) + (int(b[i + 2, 1]) << 8)
            if skip > 2147483647:
                skip -= 4294967296
            diff += skip; i += 3
        code = int(b[i, 1]) >> 2
        diff += int(b[i, 0]) + 256 * (int(b[i, 1]) & 3)
        i += 1
        if code == 0 and diff == 0:      # EOF marker
            break
        total += diff
        note = ""
        # extra fields
        while i < n and (int(b[i, 1]) >> 2) > 59:
            c2 = int(b[i, 1]) >> 2
            if c2 in (60, 61, 62):       # NUM / SUB / CHAN: one word
                i += 1
            elif c2 == 63:               # AUX note
                ln = int(b[i, 0]); nw = (ln + 1) // 2
                raw = b[i + 1: i + 1 + nw].flatten()[:ln]
                note = "".join(chr(x) for x in raw)
                i += 1 + nw
            else:
                i += 1
        sample.append(total); symbol.append(_SYM.get(code, "?")); aux.append(note)
    return np.asarray(sample, np.int64), np.asarray(symbol), np.asarray(aux)


if __name__ == "__main__":
    import sys
    base = sys.argv[1]
    h = read_header(base); x = read_signal_212(base, h); s, sym, aux = read_annotation(base)
    print(h["names"], x.shape, x[:3], "annotations:", len(s))
    u, c = np.unique(sym, return_counts=True); print(dict(zip(u, c)))
    print("rhythm notes:", sorted(set(a for a in aux if a)))
