#!/usr/bin/env bash
# Reproduces the experiments in order. Run from the experiments/ folder.
# Every step is resumable: finished (configuration, seed) pairs are skipped.
# Total compute: roughly 6-8 CPU-hours on two cores. Peak memory about 1.5 GB per worker.
set -e
export ECG_WORK="${ECG_WORK:-$(pwd)/work}"
mkdir -p "$ECG_WORK/data"

# 0. data (PhysioNet, open access; about 1 GB without PTB-XL)
cd "$ECG_WORK/data"
for db in mitdb svdb incartdb; do
  [ -d "$db/1.0.0" ] || wget -q -r -N -c -np -nH --cut-dirs=1 "https://physionet.org/files/$db/1.0.0/" -R "index.html*"
done
cd - > /dev/null

python build_caches.py                       # 1. window caches (MIT-BIH, SVDB, INCART, pooled)
python new_classic.py 4 8 16 32 64 128 256   # 2. wavelet / DCT coder on MIT-BIH (DS1-only protocol)
python new_classic_pool.py 4 8 16 32 64 128 256   #    wavelet coder on the pooled cache

python new_runner.py 1      # 3. DS1-only protocol: Transformer encoder, widths, ablations, single lead
python new_runner2.py 1     # 4. DS1-only: matched wavelet classifiers, autoencoders (needs caches from step 2)
python new_runner3.py 1     #    DS1-only: matched wavelet classifiers at 4, 8, 16 bytes
python new_runner4.py 1     # 5. three-database and leave-one-database-out training (light encoder)
python new_runner5.py 1     # 6. three-database: width sweep, ablations, single lead, matched wavelet, PTB-XL*
python new_runner6.py 1     #    three-database: extra seeds for the key comparisons
# * PTB-XL jobs are skipped unless caches/ptbxl_full.npz exists (python rev_ptbxl.py "$ECG_WORK/data" caches/ptbxl_full.npz)

python analyze.py "$ECG_WORK/res" caches "$ECG_WORK/paper_old"   # 7. DS1-only tables and numbers
python new_feats.py && python new_eval.py && python new_eval_a2.py   # 8. DS1-only evaluations
python new_eval2.py                           # 9. three-database evaluations -> work/new_eval2.json
mkdir -p "$ECG_WORK/paper/figs"
python new_analyze2.py "$ECG_WORK/paper"      # 10. tables, figures and numbers of the manuscript
python new_figs_static.py && python new_figs_extra.py   # 11. remaining figures
echo "Done. Tables and figures are in $ECG_WORK/paper"
