"""Resumable multi-seed PTB-XL trainer (12-lead, official folds), Gap 2.
Binary NORM-vs-abnormal; SemanticEncoder in_channels=12. One seed per run set,
resumable via checkpoint; appends results/ptbxl_perseed.json.
Usage: python train_ptbxl.py <seed> <total_epochs> [budget_s] [batch]
"""
import sys, os, json, time, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from config import EncoderConfig
from semantic_encoder import SemanticEncoder, SemanticDecoder, sempq_loss
seed=int(sys.argv[1]); TOTAL=int(sys.argv[2]) if len(sys.argv)>2 else 6
BUDGET=float(sys.argv[3]) if len(sys.argv)>3 else 34.0; BATCH=int(sys.argv[4]) if len(sys.argv)>4 else 64
torch.set_num_threads(os.cpu_count()); t0=time.time()
d=np.load("results/ptbxl_cache.npz"); X,y,fold=d["X"],d["y"],d["fold"]
tr=np.where(fold<=8)[0]; va=np.where(fold==9)[0]; te=np.where(fold==10)[0]
cfg=EncoderConfig(in_channels=X.shape[1], num_classes=2)
enc,dec=SemanticEncoder(cfg),SemanticDecoder(cfg); opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=3e-3)
ck="results/ckptbxl_%d.pt"%seed; start=0
if os.path.exists(ck):
    st=torch.load(ck); enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"]); opt.load_state_dict(st["opt"]); start=st["epoch"]; print("resumed",start)
torch.manual_seed(100+seed); loader=DataLoader(TensorDataset(torch.tensor(X[tr]),torch.tensor(y[tr])),batch_size=BATCH,shuffle=True)
ep=start
while ep<TOTAL:
    if time.time()-t0>BUDGET: break
    enc.train();dec.train()
    for xb,yb in loader:
        z,ee,rate=enc(xb); lo,_=sempq_loss(dec(z),ee,z,rate,yb,cfg); opt.zero_grad(); lo.backward(); opt.step()
    ep+=1; torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),"opt":opt.state_dict(),"epoch":ep},ck)
    print("seed %d epoch %d/%d (%.1fs)"%(seed,ep,TOTAL,time.time()-t0))
    if time.time()-t0+(time.time()-t0)/(ep-start)>BUDGET: break
if ep<TOTAL: print("PROGRESS %d/%d"%(ep,TOTAL)); sys.exit(0)
enc.eval();dec.eval()
def infer(idx):
    o=[]
    with torch.no_grad():
        for i in range(0,len(idx),256): o.append(dec(enc(torch.tensor(X[idx[i:i+256]]))[0]).softmax(-1).numpy())
    return np.concatenate(o,0)
pt=infer(te); s=pt[:,1]
res={"auroc":float(roc_auc_score(y[te],s)),"auprc":float(average_precision_score(y[te],s)),
     "macro_f1":float(f1_score(y[te],pt.argmax(1),average="macro"))}
p="results/ptbxl_perseed.json"; acc=json.load(open(p)) if os.path.exists(p) else {}
acc[str(seed)]=res; acc["_meta"]={"dataset":"PTB-XL 12-lead NORM-vs-abnormal","split":"official folds 1-8/9/10",
    "n_train":int(len(tr)),"n_val":int(len(va)),"n_test":int(len(te)),"epochs":TOTAL}
json.dump(acc,open(p,"w"),indent=2)
try: os.remove(ck)
except OSError: pass
print("PTB-XL seed %d FINAL:"%seed, {k:round(v,3) for k,v in res.items()})
