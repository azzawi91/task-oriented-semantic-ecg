import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="figures"
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})
pr=json.load(open("results/proof.json"))
tn=pr["task_native_decoder"]; pc=pr["pca_logreg"]; cf=pr["classical_features"]; lc=pr["learned_codecs"]
fig,ax=plt.subplots(figsize=(6.6,4.6))
ax.plot([r["bytes"] for r in tn],[r["auroc"] for r in tn],'o-',color="#c0504d",lw=2.4,ms=8,label="Task-oriented latent (ours)",zorder=5)
cae=lc["Conv autoencoder (recon)"]; js=lc["Deep-JSCC (recon+AWGN)"]
ax.plot([r["bytes"] for r in cae],[r["auroc"] for r in cae],'D-',color="#7a5195",lw=1.8,ms=6,label="Conv autoencoder (reconstruction)")
ax.plot([r["bytes"] for r in js],[r["auroc"] for r in js],'v-',color="#ef9b20",lw=1.8,ms=6,label="Deep-JSCC (reconstruction+AWGN)")
ax.plot([r["bytes"] for r in pc],[r["auroc"] for r in pc],'^-',color="#4f6d8c",lw=1.8,ms=6,label="PCA transform coding")
ax.plot([cf["bytes"]],[cf["auroc"]],'*',color="#4a8c5a",ms=16,label="Hand-crafted features",zorder=5)
ax.axhline(0.5,ls=":",color="gray",lw=1); ax.text(2.2,0.515,"chance",fontsize=8,color="gray")
ax.set_xscale("log"); ax.set_xlabel("Transmitted payload per decision (bytes, log scale)")
ax.set_ylabel("AUROC (de Chazal DS2 test)"); ax.set_ylim(0.45,0.9)
ax.set_title("Task-oriented vs reconstruction-oriented coding at matched payload")
ax.legend(loc="center right",fontsize=8)
plt.tight_layout(); plt.savefig(OUT+"/fig6_novelty_proof.png"); plt.close()
print("wrote fig6 (with learned codecs)")
print("conv-AE:", [(r["bytes"],r["auroc"]) for r in cae])
print("jscc:", [(r["bytes"],r["auroc"]) for r in js])
