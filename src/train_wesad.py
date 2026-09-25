"""WESAD multimodal stress detection: ECG-only vs PPG-only vs ECG+PPG fused.
Subject-disjoint split; SemanticEncoder (in_channels=1 or 2), binary stress.
One seed per call; appends results/wesad_result.json.
Usage: python train_wesad.py <seed>
"""
import sys, os, json, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score
from config import EncoderConfig
from semantic_encoder import SemanticEncoder, SemanticDecoder, sempq_loss
torch.set_num_threads(os.cpu_count())
seed=int(sys.argv[1])
d=np.load("results/wesad_cache.npz"); X,y,subj=d["X"],d["y"],d["subj"]
subs=np.array(sorted(set(subj.tolist())))
rng=np.random.default_rng(seed); rng.shuffle(subs)
n=len(subs); tr_s=set(subs[:9]); va_s=set(subs[9:12]); te_s=set(subs[12:])
tr=np.where(np.isin(subj,list(tr_s)))[0]; te=np.where(np.isin(subj,list(te_s)))[0]

def run(chans):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt=X[tr][:,chans,:]; Xe=X[te][:,chans,:]
    cfg=EncoderConfig(in_channels=len(chans), num_classes=2)
    enc,dec=SemanticEncoder(cfg),SemanticDecoder(cfg)
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=3e-3)
    loader=DataLoader(TensorDataset(torch.tensor(Xt),torch.tensor(y[tr])),batch_size=64,shuffle=True)
    for _ in range(6):
        enc.train();dec.train()
        for xb,yb in loader:
            z,ee,rate=enc(xb); lo,_=sempq_loss(dec(z),ee,z,rate,yb,cfg); opt.zero_grad(); lo.backward(); opt.step()
    enc.eval();dec.eval()
    with torch.no_grad():
        p=[]
        for i in range(0,len(Xe),256): p.append(dec(enc(torch.tensor(Xe[i:i+256]))[0]).softmax(-1).numpy())
        p=np.concatenate(p,0)
    return float(roc_auc_score(y[te],p[:,1])), float(f1_score(y[te],p.argmax(1),average="macro"))

import time as _t; _t0=_t.time()
p="results/wesad_result.json"; acc=json.load(open(p)) if os.path.exists(p) else {}
res=acc.get(str(seed),{})
for name,ch in [("ECG-only",[0]),("PPG-only",[1]),("ECG+PPG",[0,1])]:
    if name in res: print(name,"done"); continue
    if _t.time()-_t0>30: break
    a,f=run(ch); res[name]={"auroc":round(a,3),"macro_f1":round(f,3)}; print(name,res[name])
    acc[str(seed)]=res; acc["_meta"]={"task":"binary stress","split":"subject-disjoint 9/3/3","fs":64,"win_s":10}
    json.dump(acc,open(p,"w"),indent=2)
print("seed",seed,"configs:",list(res.keys()))
