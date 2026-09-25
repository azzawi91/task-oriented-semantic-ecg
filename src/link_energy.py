"""Packet-level link + energy model for a wearable ECG node.

Energy is ESTIMATED from a documented packet-level model parameterized by
published radio datasheets; it is NOT a hardware measurement. Two radios frame
the result so the conclusion does not hinge on one device:
  * Nordic nRF52840  BLE 1M PHY  (I_tx0dBm=9.65 mA, I_tx8dBm=16.4 mA, I_rx=6.0 mA)
  * TI CC2652R       IEEE 802.15.4 250 kbps (I_tx0dBm=24 mA approx, I_rx=6.9 mA)
Supply 3.0 V (CR2032). Encoder compute energy uses the measured MAC count of the
trained model and a CMSIS-NN INT8 throughput assumption for a Cortex-M4F @64 MHz.
"""
import json, math, numpy as np, torch
from config import EncoderConfig
from semantic_encoder import SemanticEncoder, SemanticDecoder

# ---- encoder compute cost (MACs) ----
cfg = EncoderConfig(in_channels=2)
enc = SemanticEncoder(cfg)
macs = None
try:
    from thop import profile
    x = torch.randn(1, 2, 1250)
    macs, params = profile(enc, inputs=(x,), verbose=False)
    macs = float(macs)
except Exception as e:
    macs = 3.0e6  # fallback estimate
# thop (above) only sees Conv1d/Linear modules and therefore misses the attention
# projections and the L x L score/context products of nn.MultiheadAttention
# (~7.5 M MACs at 125 Hz). The manuscript uses the analytic count below.
def analytic_macs(enc, in_ch, T, d=48, ff=96, latent=32, n_classes=3):
    def out_len(L, k, s, p): return (L + 2 * p - k) // s + 1
    macs = 0; w0 = enc.stem.out_channels; L = out_len(T, 7, 2, 3); macs += in_ch * w0 * 7 * L; c_in = w0
    for m in enc.convs:
        L = out_len(L, 5, 2, 2); macs += c_in * 5 * L + c_in * m.pw.out_channels * L; c_in = m.pw.out_channels
    for _ in enc.transformer.layers:
        macs += 3 * d * d * L + 2 * L * L * d + d * d * L + 2 * d * ff * L
    return macs + d * latent + d * n_classes
macs_thop = macs
macs = float(analytic_macs(enc, 2, 1250))
print("MACs: thop (conv+linear only) %.2fM ; analytic incl. attention %.2fM" % (macs_thop / 1e6, macs / 1e6))
M4F_MACS_PER_S = 40e6   # CMSIS-NN INT8 on Cortex-M4F @64 MHz (conservative; 100 and 200 MMAC/s also reported)
t_cpu = macs / M4F_MACS_PER_S          # s
I_cpu, V = 3.3e-3, 3.0                  # A, V
E_cpu_mJ = I_cpu * V * t_cpu * 1000.0

# ---- radio models ----
RADIOS = {
 "BLE 1M (nRF52840)":  {"R":1e6,   "I_tx":9.65e-3, "I_rx":6.0e-3, "mtu":244, "ovh":14, "ifs":150e-6, "ramp":140e-6},
 "802.15.4 (CC2652R)": {"R":250e3, "I_tx":24.0e-3, "I_rx":6.9e-3, "mtu":116, "ovh":15, "ifs":192e-6, "ramp":150e-6},
}

def tx_energy_mJ(B, r):
    n = max(1, math.ceil(B / r["mtu"]))
    phy_bits = (B + n * r["ovh"]) * 8
    t_air = phy_bits / r["R"]
    t = t_air + n * r["ifs"] + r["ramp"]
    return r["I_tx"] * V * t * 1000.0, t

PAYLOADS = {"Raw ECG window": 5000, "Semantic 8-bit": 60, "Semantic 4-bit": 44}
out = {"encoder_macs": macs, "encoder_compute_ms": t_cpu*1000, "encoder_compute_mJ": E_cpu_mJ,
       "supply_V": V, "radios": {}}
for rn, r in RADIOS.items():
    row = {}
    for pn, B in PAYLOADS.items():
        e_tx, t = tx_energy_mJ(B, r)
        e_total = e_tx + (E_cpu_mJ if pn != "Raw ECG window" else 0.0)
        row[pn] = {"bytes": B, "tx_mJ": round(e_tx, 4), "compute_mJ": round(E_cpu_mJ if pn!="Raw ECG window" else 0.0,4),
                   "total_mJ": round(e_total, 4), "airtime_ms": round(t*1000, 3)}
    raw = row["Raw ECG window"]["total_mJ"]
    for pn in PAYLOADS:
        row[pn]["energy_factor_vs_raw"] = round(raw / row[pn]["total_mJ"], 1)
    # battery life at 1 decision / 10 s on CR2032 (220 mAh, 3.0V -> 2376 J usable ~ use mWh)
    cap_mWh = 220 * 3.0  # mWh
    for pn in PAYLOADS:
        e_mWh = row[pn]["total_mJ"] / 3600.0  # mJ -> mWh (1 mWh=3600 mJ)
        decisions_per_day = 8640  # 1 per 10 s
        life_days = cap_mWh / (e_mWh * decisions_per_day) if e_mWh > 0 else 0
        row[pn]["est_battery_life_days_CR2032"] = round(life_days, 1)
    out["radios"][rn] = row

json.dump(out, open("results/link_energy.json", "w"), indent=2)
print("encoder MACs=%.2e  compute=%.2f ms  %.4f mJ" % (macs, t_cpu*1000, E_cpu_mJ))
for rn, row in out["radios"].items():
    print("\n[%s]" % rn)
    for pn in PAYLOADS:
        d = row[pn]
        print("  %-16s %5dB  tx=%.4f mJ  total=%.4f mJ  x%.1f vs raw  airtime=%.2fms  life=%.1f d" %
              (pn, d["bytes"], d["tx_mJ"], d["total_mJ"], d["energy_factor_vs_raw"], d["airtime_ms"], d["est_battery_life_days_CR2032"]))

# ---- packet-loss / ARQ retransmission regime (BLE) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def expected_energy(B, r, per):
    n = max(1, math.ceil(B / r["mtu"]))
    e1, _ = tx_energy_mJ(B, r)            # one full attempt of all packets
    # stop-and-wait ARQ: expected attempts per packet = 1/(1-per)
    return e1 / max(1e-6, (1 - per))
pers = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
sweep = {}
for rn, r in RADIOS.items():
    fac = []
    for p in pers:
        e_raw = expected_energy(5000, r, p)
        e_sem = expected_energy(60, r, p) + E_cpu_mJ
        fac.append(round(e_raw / e_sem, 2))
    sweep[rn] = fac
out["per_sweep"] = {"per": pers, "energy_factor_vs_raw": sweep}
json.dump(out, open("results/link_energy.json", "w"), indent=2)

OUT="figures"
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.5,4))
labels=list(PAYLOADS); xi=np.arange(len(labels)); w=0.35
for i,(rn,r) in enumerate(RADIOS.items()):
    vals=[out["radios"][rn][p]["total_mJ"] for p in labels]
    ax1.bar(xi+(i-0.5)*w,vals,w,label=rn.split(" (")[0],edgecolor="black",lw=0.4)
ax1.set_yscale("log"); ax1.set_xticks(xi); ax1.set_xticklabels([l.replace(" ","\n") for l in labels],fontsize=8)
ax1.set_ylabel("Energy per decision (mJ, log)"); ax1.set_title("(a) Per-decision energy (compute + radio)")
ax1.legend(fontsize=8)
for rn in RADIOS:
    ax2.plot([p*100 for p in pers],sweep[rn],'o-',lw=2,ms=6,label=rn.split(" (")[0])
ax2.set_xlabel("Packet error rate (%)"); ax2.set_ylabel("Energy reduction factor vs raw")
ax2.set_title("(b) Semantic advantage grows under loss"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT+"/fig7_link_energy.png"); plt.close()
print("\nPER sweep (energy factor vs raw):")
for rn in RADIOS: print("  %-20s"%rn, dict(zip(pers,sweep[rn])))
print("wrote fig7_link_energy.png")
