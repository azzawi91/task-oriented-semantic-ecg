"""Benchmark the post-quantum-ready security layer on this CPU.
ML-KEM-512 (FIPS 203) and ML-DSA-44/Dilithium-2 (FIPS 204) via pure-Python
reference implementations (kyber-py, dilithium-py): exact standardized sizes and
real x86 reference-implementation timing. AES-128-GCM via the `cryptography`
library on the 60-byte semantic payload. Reference-implementation timings are
not optimized C and over-state a deployed system; published pqm4 Cortex-M4
cycle counts are cited in the manuscript for the embedded target.
Writes results/pq_bench.json.
"""
import json, time, statistics, os
import numpy as np
from kyber_py.ml_kem import ML_KEM_512
from dilithium_py.ml_dsa import ML_DSA_44
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def bench(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000.0)
    return {"ms_mean": round(statistics.mean(ts), 3), "ms_std": round(statistics.pstdev(ts), 3), "n": n}

out = {"backend": "pure-Python reference (kyber-py / dilithium-py), x86 CPU"}
# ML-KEM-512
ek, dk = ML_KEM_512.keygen()
K, ct = ML_KEM_512.encaps(ek)
out["ML-KEM-512"] = {
    "ek_bytes": len(ek), "dk_bytes": len(dk), "ct_bytes": len(ct), "ss_bytes": len(K),
    "keygen": bench(lambda: ML_KEM_512.keygen(), 20),
    "encaps": bench(lambda: ML_KEM_512.encaps(ek), 20),
    "decaps": bench(lambda: ML_KEM_512.decaps(dk, ct), 20),
}
# ML-DSA-44
pk, sk = ML_DSA_44.keygen()
msg = b"semantic-latent-60B-payload"
sig = ML_DSA_44.sign(sk, msg)
out["ML-DSA-44"] = {
    "pk_bytes": len(pk), "sig_bytes": len(sig),
    "keygen": bench(lambda: ML_DSA_44.keygen(), 8),
    "sign":   bench(lambda: ML_DSA_44.sign(sk, msg), 8),
    "verify": bench(lambda: ML_DSA_44.verify(pk, msg, sig), 8),
}
# AES-128-GCM on the 60-byte payload (representative, optimized C backend)
key = os.urandom(16); aead = AESGCM(key); nonce = os.urandom(12); pt = os.urandom(60)
ctx = aead.encrypt(nonce, pt, b"")
out["AES-128-GCM_60B"] = {
    "encrypt": bench(lambda: aead.encrypt(nonce, pt, b""), 2000),
    "decrypt": bench(lambda: aead.decrypt(nonce, ctx, b""), 2000),
    "ciphertext_bytes": len(ctx), "overhead_bytes": len(ctx) - len(pt) + 12,
}
out["handshake_total_bytes"] = len(ek) + len(ct) + len(pk) + len(sig)
json.dump(out, open("results/pq_bench.json", "w"), indent=2)
print("ML-KEM-512:", out["ML-KEM-512"]["ek_bytes"], out["ML-KEM-512"]["ct_bytes"],
      "| keygen", out["ML-KEM-512"]["keygen"]["ms_mean"], "encaps", out["ML-KEM-512"]["encaps"]["ms_mean"], "decaps", out["ML-KEM-512"]["decaps"]["ms_mean"], "ms")
print("ML-DSA-44: pk", out["ML-DSA-44"]["pk_bytes"], "sig", out["ML-DSA-44"]["sig_bytes"],
      "| keygen", out["ML-DSA-44"]["keygen"]["ms_mean"], "sign", out["ML-DSA-44"]["sign"]["ms_mean"], "verify", out["ML-DSA-44"]["verify"]["ms_mean"], "ms")
print("AES-128-GCM 60B: enc", out["AES-128-GCM_60B"]["encrypt"]["ms_mean"], "dec", out["AES-128-GCM_60B"]["decrypt"]["ms_mean"], "ms; handshake total", out["handshake_total_bytes"], "B")
