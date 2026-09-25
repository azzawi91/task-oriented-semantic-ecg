"""Gap-4 extended novelty proof: task-oriented latent vs learned reconstruction
codecs (conv-AE, deep-JSCC) vs PCA vs hand-crafted features, at matched payload,
de Chazal DS2, all read by ONE shared logistic-regression classifier.
Appends to results/proof.json under 'learned_codecs'.
"""
import json, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import metrics as M, dechazal as DC
from config import EncoderConfig
from semantic_encoder import SemanticEncoder

d = np.load("results/mitbih_cache.npz", allow_pickle=True)
X, y, g = d["X"], d["y"], d["groups"]
tr, va, te = DC.split(y, g, seed=0)
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
C, T, D = X.shape[1], X.shape[2], 32

class ConvAEEncoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(C,16,7,stride=4,padding=3), nn.GELU(),
            nn.Conv1d(16,32,5,stride=4,padding=2), nn.GELU(),
            nn.Conv1d(32,48,5,stride=4,padding=2), nn.GELU())
        self.fc = nn.Linear(48*20, D)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

def quant(z, bits):
    qmax = max(1, 2**(bits-1)-1)
    return np.clip(np.round(z*qmax), -qmax-1, qmax)/qmax

def logreg_auroc(Ftr, Fte):
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0).fit(sc.transform(Ftr), ytr)
    p = clf.predict_proba(sc.transform(Fte)); full = np.zeros((len(Fte),3))
    for j,cc in enumerate(clf.classes_): full[:,int(cc)] = p[:,j]
    return M.auroc_multiclass(yte, full)

# task latent (SemanticEncoder, model_dechazal_0)
st = torch.load("results/model_dechazal_0.pt"); te_cfg = EncoderConfig(in_channels=st["cfg_in"])
tenc = SemanticEncoder(te_cfg); tenc.load_state_dict(st["enc"]); tenc.eval()
def task_latent(Xa):
    o=[]
    with torch.no_grad():
        for i in range(0,len(Xa),256):
            x=torch.tensor(Xa[i:i+256]); h=tenc.stem(x); h=tenc.convs(h); h=h.transpose(1,2)
            h=tenc.transformer(h); o.append(tenc.proj_latent(h.mean(1)).numpy())
    return np.concatenate(o,0)

def ae_latent(model_path):
    s=torch.load(model_path); e=ConvAEEncoder(C,D); e.load_state_dict(s["enc"]); e.eval()
    o=[]
    with torch.no_grad():
        for Xa in (Xtr,Xte):
            oo=[]
            for i in range(0,len(Xa),256):
                oo.append(e(torch.tensor(Xa[i:i+256])).numpy())
            o.append(np.concatenate(oo,0))
    return o  # [train_lat, test_lat]

reps = {"Task-oriented latent": [task_latent(Xtr), task_latent(Xte)],
        "Conv autoencoder (recon)": ae_latent("results/model_recon_dechazal.pt"),
        "Deep-JSCC (recon+AWGN)": ae_latent("results/model_jscc_dechazal.pt")}
bits_bytes = [(8,32),(4,16),(2,8)]
out = {}
for name,(Ltr,Lte) in reps.items():
    out[name] = [{"bytes":by,"bits":b,"auroc":round(logreg_auroc(quant(Ltr,b),quant(Lte,b)),3)} for b,by in bits_bytes]
pr = json.load(open("results/proof.json")); pr["learned_codecs"]=out
json.dump(pr, open("results/proof.json","w"), indent=2)
for name in out:
    print(name, "->", [(r["bytes"],r["auroc"]) for r in out[name]])
