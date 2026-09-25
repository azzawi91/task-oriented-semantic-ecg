"""Novelty proof: task-oriented coding vs reconstruction-oriented (PCA) coding
at matched payload, on the de Chazal DS1/DS2 partition.

All methods transmit a fixed-size quantized code and are classified by the SAME
downstream logistic-regression head, so differences reflect the representation
(its training objective), not classifier capacity. The task model's native
trained decoder is also reported as the deployed operating point.
Writes results/proof.json.
"""
import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import metrics as M
import dechazal as DC
from config import EncoderConfig
from semantic_encoder import SemanticEncoder, SemanticDecoder

d = np.load("results/mitbih_cache.npz", allow_pickle=True)
X, y, g = d["X"], d["y"], d["groups"]
tr, va, te = DC.split(y, g, seed=0)
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
D = 32

st = torch.load("results/model_dechazal_0.pt")
cfg = EncoderConfig(in_channels=st["cfg_in"])
enc, dec = SemanticEncoder(cfg), SemanticDecoder(cfg)
enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"]); enc.eval(); dec.eval()

def latent(Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 256):
            x = torch.tensor(Xa[i:i+256])
            h = enc.stem(x); h = enc.convs(h); h = h.transpose(1, 2)
            h = enc.transformer(h); pooled = h.mean(1)
            out.append(enc.proj_latent(pooled).numpy())
    return np.concatenate(out, 0)

def task_decode(zq):
    with torch.no_grad():
        return dec(torch.tensor(zq, dtype=torch.float32)).softmax(-1).numpy()

def quant(z, bits):
    qmax = max(1, 2 ** (bits - 1) - 1)
    return np.clip(np.round(z * qmax), -qmax - 1, qmax) / qmax

def logreg_auroc(Ftr, Fte):
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    clf.fit(sc.transform(Ftr), ytr)
    p = clf.predict_proba(sc.transform(Fte))
    full = np.zeros((p.shape[0], 3))
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = p[:, j]
    return M.auroc_multiclass(yte, full), M.macro_f1(yte, full.argmax(1))

Ztr, Zte = latent(Xtr), latent(Xte)

# --- Task-oriented latent (common logreg classifier) ---
task_logreg = []
for b in [8, 6, 4, 3, 2]:
    a, f = logreg_auroc(quant(Ztr, b), quant(Zte, b))
    task_logreg.append({"bytes": int(np.ceil(D * b / 8)), "bits": b, "auroc": a, "macro_f1": f})

# --- Task-oriented latent (native trained decoder) ---
task_native = []
for b in [8, 6, 4, 3, 2]:
    p = task_decode(quant(Zte, b))
    task_native.append({"bytes": int(np.ceil(D * b / 8)), "bits": b,
                        "auroc": M.auroc_multiclass(yte, p), "macro_f1": M.macro_f1(yte, p.argmax(1))})

# --- PCA transform coding (8-bit coeffs) + common logreg ---
Ftr = Xtr.reshape(len(Xtr), -1); Fte = Xte.reshape(len(Xte), -1)
pca = PCA(n_components=128, random_state=0).fit(Ftr)
Ptr_all, Pte_all = pca.transform(Ftr), pca.transform(Fte)
scale = np.abs(Ptr_all).max(0) + 1e-9
pca_curve = []
for k in [2, 4, 8, 16, 32, 64, 128]:
    qt = np.round(Ptr_all[:, :k] / scale[:k] * 127) / 127 * scale[:k]
    qe = np.round(Pte_all[:, :k] / scale[:k] * 127) / 127 * scale[:k]
    a, f = logreg_auroc(qt, qe)
    pca_curve.append({"bytes": k, "k": k, "auroc": a, "macro_f1": f})

out = {"protocol": "de Chazal DS1/DS2, seed 0", "n_train": int(len(tr)), "n_test": int(len(te)),
       "task_latent_logreg": task_logreg, "task_native_decoder": task_native, "pca_logreg": pca_curve}
json.dump(out, open("results/proof.json", "w"), indent=2)
print("=== Task latent + common logreg ==="); [print("  %2dB  AUROC %.3f" % (r["bytes"], r["auroc"])) for r in task_logreg]
print("=== Task native decoder ==="); [print("  %2dB  AUROC %.3f" % (r["bytes"], r["auroc"])) for r in task_native]
print("=== PCA + common logreg ==="); [print("  %3dB  AUROC %.3f" % (r["bytes"], r["auroc"])) for r in pca_curve]
