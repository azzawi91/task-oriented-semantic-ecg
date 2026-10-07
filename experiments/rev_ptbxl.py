"""PTB-XL (full, 21,799 records, 100 Hz) download + cache builder (Reviewer 1 minor #1).

Downloads the official PhysioNet zip (~1.7 GB) once, extracts only the 100 Hz
records and the two CSVs, and builds a cache with
  X [N, 12, 1000] per-record per-lead z-scored, y = 0 (NORM only) / 1 (any other
  diagnostic superclass), groups = official strat_fold as strings.
Records without any diagnostic superclass are dropped (as in the original code).

Usage: python rev_ptbxl.py <data_dir> <out.npz>
"""
from __future__ import annotations
import ast, csv, os, sys, time, urllib.request, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wfdb_lite as W

ZIP_URL = ("https://physionet.org/static/published-projects/ptb-xl/"
           "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")


def ensure_data(data_dir, log=print):
    root = os.path.join(data_dir, "ptbxl")
    if os.path.exists(os.path.join(root, "ptbxl_database.csv")) and os.path.isdir(os.path.join(root, "records100")):
        return root
    os.makedirs(data_dir, exist_ok=True)
    zpath = os.path.join(data_dir, "ptbxl_1.0.3.zip")
    if not os.path.exists(zpath):
        log("  downloading PTB-XL zip (~1.8 GB, resumable) ...")
        part = zpath + ".part"; have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(ZIP_URL, headers={"Range": "bytes=%d-" % have} if have else {})
        with urllib.request.urlopen(req, timeout=120) as r:
            total = have + int(r.headers.get("Content-Length", "0") or 0)
            if have and r.status != 206:                     # server ignored the range -> start over
                have = 0; mode = "wb"
            else:
                mode = "ab" if have else "wb"
            with open(part, mode) as f:
                done = have; last = time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk); done += len(chunk)
                    if time.time() - last > 60:
                        log("    %.0f / %.0f MB" % (done / 1e6, total / 1e6)); last = time.time()
        os.rename(part, zpath)
    log("  extracting 100 Hz records ...")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist(); top = names[0].split("/")[0]
        want = [n for n in names if ("/records100/" in n or n.endswith("ptbxl_database.csv") or n.endswith("scp_statements.csv"))]
        for i, n in enumerate(want):
            if n.endswith("/"):
                continue
            dst = os.path.join(root, n[len(top) + 1:])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as f:
                f.write(src.read())
            if i % 5000 == 0:
                log("    %d / %d files" % (i, len(want)))
    return root


def build_cache(root, out, log=print):
    with open(os.path.join(root, "scp_statements.csv")) as f:
        rows = list(csv.DictReader(f))
    def _is_diag(v):                                     # the CSV stores 1.0 / "" (float-formatted)
        try:
            return float(v) == 1.0
        except (TypeError, ValueError):
            return False
    diag = {r[""] if "" in r else r[list(r.keys())[0]]: r["diagnostic_class"] for r in rows if _is_diag(r.get("diagnostic"))}
    X, y, fold, ids = [], [], [], []
    with open(os.path.join(root, "ptbxl_database.csv")) as f:
        for i, r in enumerate(csv.DictReader(f)):
            scp = ast.literal_eval(r["scp_codes"]); sup = {diag[k] for k in scp if k in diag}
            if not sup:
                continue
            base = os.path.join(root, r["filename_lr"])
            try:
                sig, hdr = W.read_signal(base)              # [1000, 12]
            except Exception as e:                           # noqa
                log("  [skip] %s: %s" % (r["filename_lr"], e)); continue
            sig = np.nan_to_num(sig.T.astype(np.float32))
            for c in range(sig.shape[0]):
                s = sig[c]; sd = s.std(); sig[c] = (s - s.mean()) / sd if sd > 1e-6 else s - s.mean()
            X.append(sig); y.append(0 if sup == {"NORM"} else 1); fold.append(str(int(r["strat_fold"]))); ids.append(int(r["ecg_id"]))
            if len(X) % 2000 == 0:
                log("  loaded %d" % len(X))
    X = np.stack(X); y = np.asarray(y, np.int64); fold = np.asarray(fold); ids = np.asarray(ids)
    np.savez_compressed(out, X=X, y=y, groups=fold, ecg_id=ids, fs=100, leads="12")
    log("[cache] PTB-XL %s NORM/abnormal = %d/%d; folds %s" % (X.shape, int((y == 0).sum()), int((y == 1).sum()),
                                                                dict(zip(*[a.tolist() for a in np.unique(fold, return_counts=True)]))))
    return out


if __name__ == "__main__":
    root = ensure_data(sys.argv[1]); build_cache(root, sys.argv[2])
