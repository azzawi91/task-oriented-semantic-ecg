#!/usr/bin/env bash
# Full 10-seed de Chazal (DS1/DS2) run on CPU. No GPU needed (~50k-parameter model).
# Typical laptop CPU: roughly 1.5-2 min per seed -> ~15-20 min total.
# Resumable: if interrupted, just run again; finished seeds are skipped.
#
# Usage:
#   bash reproduce_dechazal_10seeds.sh /path/to/mitdb/1.0.0
# If you omit the path it looks for ./physionet.org/files/mitdb/1.0.0
set -e
cd "$(dirname "$0")"
MITDB="${1:-./physionet.org/files/mitdb/1.0.0}"
pip install --quiet wfdb scipy scikit-learn torch numpy 2>/dev/null || true

# 1) Build the windowed cache once (from the raw MIT-BIH records)
if [ ! -f results/mitbih_cache.npz ]; then
  echo "[1/3] building window cache from $MITDB"
  python3 - "$MITDB" <<'PY'
import sys, os, numpy as np, data_mitbih as dmb
X,y,g = dmb.build_dataset(sys.argv[1], dmb.MITBIHConfig(target_fs=125))
os.makedirs("results", exist_ok=True)
np.savez_compressed("results/mitbih_cache.npz", X=X, y=y, groups=g)
print("cache:", X.shape)
PY
fi

# 2) Train all 10 seeds to completion (6 epochs each; huge time budget = finish in one pass)
echo "[2/3] training 10 seeds (resumable)"
for s in 0 1 2 3 4 5 6 7 8 9; do
  python3 train_dechazal.py "$s" 6 1000000 64
done

# 3) Aggregate
echo "[3/3] aggregate over all seeds"
python3 - <<'PY'
import json, numpy as np
d = json.load(open("results/full_dechazal_perseed.json"))
seeds = [k for k in d if k != "_meta"]
print("seeds:", sorted(int(s) for s in seeds))
for k in ["auroc","auprc","macro_f1","sens_at_spec","f1_at_op"]:
    v = [d[s][k] for s in seeds]
    print("  %-13s %.3f +/- %.3f  [%.3f, %.3f]" % (k, np.mean(v), np.std(v), min(v), max(v)))
PY
echo "Done. Per-seed metrics in results/full_dechazal_perseed.json"
