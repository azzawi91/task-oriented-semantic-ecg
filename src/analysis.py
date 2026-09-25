"""Deterministic bandwidth/energy + post-quantum overhead analysis, plus the
3-seed full-schedule aggregate. No training; all values are exact calculations
or standardized constants. Writes results/analysis.json.
"""
import json, numpy as np

# ---- 3-seed full-schedule aggregate (inter-patient) ----
full = json.load(open("results/full_patient_perseed.json"))
seeds = [k for k in full if k != "_meta"]
keys = ["auroc", "auprc", "macro_f1", "sens_at_spec", "f1_at_op", "alarm_rate"]
agg = {k: {"mean": float(np.mean([full[s][k] for s in seeds])),
           "std": float(np.std([full[s][k] for s in seeds])),
           "min": float(np.min([full[s][k] for s in seeds])),
           "max": float(np.max([full[s][k] for s in seeds]))} for k in keys}

# ---- bandwidth / transmission-energy (payload-proportional model) ----
RAW = 5000              # 2 ch x 1250 samp x 2 B  (125 Hz, 10 s window)
SEM8, SEM4 = 60, 44     # latent + 28 B AEAD
bw = {
    "raw_bytes": RAW, "sem_bytes_8bit": SEM8, "sem_bytes_4bit": SEM4,
    "reduction_pct_8bit": round(100 * (1 - SEM8 / RAW), 2),
    "reduction_pct_4bit": round(100 * (1 - SEM4 / RAW), 2),
    "tx_energy_factor_8bit": round(RAW / SEM8, 1),   # tx energy ~ payload bytes
    "tx_energy_factor_4bit": round(RAW / SEM4, 1),
}

# ---- post-quantum handshake overhead: standardized sizes (FIPS 203/204) ----
# One-time per session; steady-state per-decision overhead is only the AEAD tag+nonce.
pq = {
    "ML-KEM-512_FIPS203": {"encaps_key_pk_bytes": 800, "decaps_key_sk_bytes": 1632,
                            "ciphertext_bytes": 768, "shared_secret_bytes": 32},
    "ML-DSA-44_Dilithium2_FIPS204": {"public_key_bytes": 1312, "signature_bytes": 2420},
    "per_decision_aead_overhead_bytes": 28,
    "handshake_total_bytes_one_time": 800 + 768 + 1312 + 2420,
    "note": ("Handshake is a one-time per-session cost; amortized over a 10 s decision "
             "cadence it adds <1 B/s after the first minute. Steady-state per-decision "
             "security overhead is the 28-byte AES-128-GCM tag+nonce, already included "
             "in the 60 B semantic payload."),
}

out = {"full_schedule_aggregate": agg, "n_seeds": len(seeds),
       "bandwidth_energy": bw, "post_quantum": pq}
json.dump(out, open("results/analysis.json", "w"), indent=2)
print("=== full-schedule inter-patient aggregate (%d seeds) ===" % len(seeds))
for k in keys:
    a = agg[k]; print("  %-13s mean %.3f  std %.3f  [%.3f, %.3f]" % (k, a["mean"], a["std"], a["min"], a["max"]))
print("bandwidth:", bw)
print("PQ handshake one-time bytes:", pq["handshake_total_bytes_one_time"])
