"""PTB-XL loader (second-dataset / multi-channel experiment, Gap 2).

PTB-XL: 21,837 clinical 12-lead ECGs, 10 s, 100 Hz, with official stratified
folds (strat_fold 1-8 train, 9 validation, 10 test) [Wagner 2020]. This loader
builds 12-lead 10 s windows and a binary NORM-vs-abnormal label from the
diagnostic superclasses, exercising the encoder's multi-channel input.
"""
import os, ast, numpy as np, pandas as pd, wfdb

def build_cache(root, out="results/ptbxl_cache.npz", fs=100, max_records=None, verbose=True):
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    agg = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]
    def superclass(scp):
        d = ast.literal_eval(scp); out = set()
        for k in d:
            if k in agg.index: out.add(agg.loc[k, "diagnostic_class"])
        return out
    df["sup"] = df.scp_codes.apply(superclass)
    fcol = "filename_lr" if fs == 100 else "filename_hr"
    X, y, fold = [], [], []
    rows = df.itertuples()
    n = 0
    for r in rows:
        sup = r.sup
        if not sup: continue
        rec = os.path.join(root, getattr(r, fcol))
        try:
            sig, _ = wfdb.rdsamp(rec)
        except Exception:
            continue
        sig = np.nan_to_num(sig.T.astype(np.float32))   # [12, T]
        for c in range(sig.shape[0]):
            s = sig[c]; sd = s.std(); sig[c] = (s - s.mean())/sd if sd > 1e-6 else s - s.mean()
        X.append(sig); y.append(0 if sup == {"NORM"} else 1); fold.append(int(r.strat_fold))
        n += 1
        if max_records and n >= max_records: break
        if verbose and n % 1000 == 0: print("loaded", n)
    X = np.stack(X); y = np.array(y, np.int64); fold = np.array(fold, np.int64)
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(out, X=X, y=y, fold=fold)
    print("PTB-XL cache:", X.shape, "NORM/abn:", int((y==0).sum()), int((y==1).sum()))
    return out

if __name__ == "__main__":
    import sys
    build_cache(sys.argv[1], max_records=int(sys.argv[2]) if len(sys.argv) > 2 else None)
