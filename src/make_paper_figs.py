import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT="figures"
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})

sw=json.load(open("results/sweeps.json"))
an=json.load(open("results/analysis.json"))
logp=json.load(open("results/mitbih_metrics_logreg_patient.json"))
logr=json.load(open("results/mitbih_metrics_logreg_random.json"))
agg=an["full_schedule_aggregate"]

# ---- Fig: rate-fidelity Pareto ----
rf=sw["rate_fidelity"]
bytes_=[r["payload_bytes"] for r in rf]; auc=[r["auroc"] for r in rf]; bits=[r["bits"] for r in rf]
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(bytes_,auc,'o-',color="#c0504d",lw=2,ms=7,label="Semantic latent (ours)")
for x,yv,b in zip(bytes_,auc,bits): ax.annotate(f"{b}-bit",(x,yv),textcoords="offset points",xytext=(6,6),fontsize=8)
ax.axvline(sw["raw_bytes_per_decision"],ls="--",color="gray",lw=1.2)
ax.annotate(f"raw ECG = {sw['raw_bytes_per_decision']} B",(sw['raw_bytes_per_decision'],0.62),
            xytext=(-150,0),textcoords="offset points",fontsize=9,color="gray",
            arrowprops=dict(arrowstyle="->",color="gray"))
ax.set_xscale("log"); ax.set_xlabel("Payload per decision (bytes, log scale)")
ax.set_ylabel("AUROC (inter-patient test)"); ax.set_ylim(0.5,0.9)
ax.set_title("Rate–fidelity trade-off: diagnostic AUROC vs transmitted payload")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_rate_fidelity.png"); plt.close()

# ---- Fig: packet-loss robustness ----
pl=sw["packet_loss"]
x=[r["loss_rate"]*100 for r in pl]; m=[r["auroc_mean"] for r in pl]; s=[r["auroc_std"] for r in pl]
fig,ax=plt.subplots(figsize=(6,4))
ax.errorbar(x,m,yerr=s,fmt='s-',color="#5b7c99",lw=2,ms=6,capsize=3)
ax.fill_between(x,np.array(m)-np.array(s),np.array(m)+np.array(s),color="#5b7c99",alpha=0.15)
ax.set_xlabel("Latent-coefficient erasure / packet-loss rate (%)")
ax.set_ylabel("AUROC (inter-patient test)"); ax.set_ylim(0.7,0.88)
ax.set_title("Graceful degradation under channel loss (10 random masks/point)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_packet_loss.png"); plt.close()

# ---- Fig: model comparison ----
groups=["LogReg\n(random)","LogReg\n(inter-patient)","Semantic encoder\n(inter-patient)"]
metrics=[("auroc","AUROC"),("macro_f1","Macro-F1")]
vals={"auroc":[logr["auroc"]["mean"],logp["auroc"]["mean"],agg["auroc"]["mean"]],
      "macro_f1":[logr["macro_f1"]["mean"],logp["macro_f1"]["mean"],agg["macro_f1"]["mean"]]}
errs={"auroc":[0,0,agg["auroc"]["std"]],"macro_f1":[0,0,agg["macro_f1"]["std"]]}
xi=np.arange(3); w=0.35
fig,ax=plt.subplots(figsize=(6.5,4))
for i,(mk,mn) in enumerate(metrics):
    ax.bar(xi+(i-0.5)*w,vals[mk],w,yerr=errs[mk],capsize=3,label=mn,
           color=["#9aa7b1","#c0504d"][i],edgecolor="black",lw=0.4)
ax.set_xticks(xi); ax.set_xticklabels(groups,fontsize=9); ax.set_ylim(0,0.9)
ax.set_ylabel("Score"); ax.axhline(1/3,ls="--",lw=0.7,color="gray")
ax.annotate("chance (3-class)",(2.1,0.34),fontsize=8,color="gray")
ax.set_title("MIT-BIH arrhythmia detection: encoder vs baselines (mean, full schedule)")
ax.legend(loc="upper left"); plt.tight_layout(); plt.savefig(f"{OUT}/fig4_model_comparison.png"); plt.close()

# ---- Fig: bandwidth/energy ----
fig,ax=plt.subplots(figsize=(6,4))
labels=["Raw ECG\n(5000 B)","Semantic 8-bit\n(60 B)","Semantic 4-bit\n(44 B)"]
v=[an["bandwidth_energy"]["raw_bytes"],an["bandwidth_energy"]["sem_bytes_8bit"],an["bandwidth_energy"]["sem_bytes_4bit"]]
bars=ax.bar(labels,v,color=["#9aa7b1","#c0504d","#8c4a48"],edgecolor="black",lw=0.4)
ax.set_yscale("log"); ax.set_ylabel("Payload per decision (bytes, log)")
ax.set_title("Per-decision payload: 98.8%% reduction vs raw transmission")
for b,val in zip(bars,v): ax.annotate(f"{val} B",(b.get_x()+b.get_width()/2,val),
    textcoords="offset points",xytext=(0,4),ha="center",fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_bandwidth.png"); plt.close()

# ---- Fig: architecture schematic ----
fig,ax=plt.subplots(figsize=(9,3.2)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3)
def box(x,y,w,h,t,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",fc=c,ec="black",lw=1.2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8.5)
def arrow(x1,y1,x2,y2,t=""):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.3,color="#333"))
    if t: ax.text((x1+x2)/2,(y1+y2)/2+0.18,t,ha="center",fontsize=7.5,color="#333")
box(0.1,1.1,1.7,0.9,"Wearable\nECG sensor\n(ECG x2)","#eaf2f8")
box(2.1,1.1,1.9,0.9,"Semantic encoder\nCNN+Transformer\n→ INT8 latent","#fdecea")
box(4.3,1.1,1.7,0.9,"PQ-Lite\nAES-GCM\n(28 B tag)","#fef5e7")
box(6.1,0.9,1.6,1.3,"Lossy wireless\nchannel\n(BAN/uplink)","#f4f6f7")
box(7.9,1.1,2.0,0.9,"Edge gateway\nsemantic decoder\n→ diagnosis","#eafaf1")
arrow(1.8,1.55,2.1,1.55); arrow(4.0,1.55,4.3,1.55,"60 B"); arrow(6.0,1.55,6.1,1.55)
arrow(7.7,1.55,7.9,1.55)
ax.text(5.0,2.7,"Task-oriented semantic communication for wearable ECG",
        ha="center",fontsize=10,weight="bold")
ax.text(2.55,0.75,"raw 5000 B never leaves device",fontsize=7,color="#a00")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_architecture.png"); plt.close()
print("figures written to",OUT)
import os; print("\n".join(sorted(os.listdir(OUT))))
