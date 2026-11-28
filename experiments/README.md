# Experiments

Run everything with `bash run_all.sh` from this folder, or follow the steps below one at a time. All scripts read and write under `work/` (change it with the environment variable `ECG_WORK`).

## Scripts, in the order they are used

| Step | Script | What it does |
|---|---|---|
| 1 | `build_caches.py` | Cuts MIT-BIH, SVDB and INCART into labelled 10-s windows at 125 Hz and builds the pooled three-database cache |
| 2 | `new_classic.py`, `new_classic_pool.py` | Wavelet (db4) and DCT coders with idealised entropy coding at 4 to 256 bytes; writes decoded copies of the caches |
| 3 | `new_runner.py` | DS1-only protocol: Transformer encoder, code widths, ablations, single lead |
| 4 | `new_runner2.py`, `new_runner3.py` | DS1-only: classifiers retrained on wavelet-decoded windows; autoencoder baselines |
| 5 | `new_runner4.py` | Three-database and leave-one-database-out training with the light encoder |
| 6 | `new_runner5.py`, `new_runner6.py` | Three-database: width sweep, loss ablations, single lead, matched wavelet classifiers, PTB-XL |
| 7 | `analyze.py` | DS1-only tables and numbers |
| 8 | `new_eval.py`, `new_eval_a2.py`, `new_feats.py` | DS1-only evaluations (wavelet coder, external databases, decoded autoencoders) |
| 9 | `new_eval2.py` | Three-database evaluations: code-size sweep, requantization, wavelet coder, leave-one-database-out, paired tests |
| 10 | `new_analyze2.py` | Tables, figures and numbers of the manuscript |
| 11 | `new_figs_static.py`, `new_figs_extra.py` | Concept figure, reconstruction example, per-seed plot, external-validation plot |

## Later experiments (run after the steps above)

| Step | Script | What it does |
|---|---|---|
| 12 | `new_runner7.py` | Seven more Transformer seeds; record-mixed control (half of the DS2 windows admitted to training) |
| 13 | `new_runner8.py` | Training sets of 18, 36 and 72 records at an equal number of training steps; first cross-validation batch |
| 14 | `new_runner9.py` | Nine more repetitions of patient-level five-fold cross-validation, each with its own partition |
| 15 | `new_stats.py ds2`, `ext`, `late`, `cvlc`; `new_lc2.py` | Paired two-level bootstrap (patients x seeds) for every comparison; writes `work/new_stats.json` |
| 16 | `new_hybrid.py`, `new_hybrid2.py` | Hybrid code (4 learned bytes + 10 RR statistics) on DS2 and the two held-out databases; 32-byte and per-class controls |
| 17 | `new_cvrep.py hyb`, `rate`, `pop` | Hybrid code, code size and training set under repeated cross-validation (fast bootstrap, `fastboot.py`) |
| 18 | `new_edb_cache.py`, `new_confirm.py`, `new_confirm2.py` | Confirmatory test on the European ST-T Database, as fixed in `../CONFIRMATION_PLAN.md` |

The European ST-T Database is expected in `work/data/edb/1.0.0` (download from https://physionet.org/files/edb/1.0.0/). The RR-feature files `caches/rr_features_fs125.npy`, `caches/rr_svdb.npy` and `caches/rr_incart.npy` are `rev_np.rr_features(X, 125)` applied to the three window caches.

## Library modules

| Module | Contents |
|---|---|
| `rev_model.py` | Encoder (light CNN or CNN + Transformer), decoder, loss, parameter and operation counts |
| `rev_train.py` | Trains one configuration and seed; exports codes, decoder weights, probabilities, metrics |
| `rev_recon.py` | Autoencoder baselines trained for reconstruction |
| `rev_data.py`, `new_xdb_cache.py`, `rev_ptbxl.py` | Window caches for MIT-BIH, SVDB/INCART, PTB-XL |
| `rev_splits.py` | DS1/DS2 partition and the other splits |
| `rev_metrics.py`, `rev_np.py` | Metrics, quantizers, hand-crafted RR-interval and spectral codes |
| `wfdb_lite.py` | Minimal reader for PhysioNet format 212 |

## Train a single model

```bash
python build_caches.py
# light encoder, 4-byte code, three databases, seed 0
python rev_train.py --cache caches/pool_all.npz --name demo --seed 0 --out work/res \
  --set split='"pool"' pool_dbs='["svdb","incart"]' use_transformer=False latent=4
```

The DS2 metrics are printed and saved to `work/res/demo/seed0.json`.

## Notes

- The runners can be started several times in parallel (`python new_runner4.py 1 &`, `python new_runner4.py 2 &`); they share work through lock folders in `work/`.
- Some configurations in the runners list ten seeds where the paper reports five or three. The seeds actually used are in `../results/per_seed_results.json`.
- File paths were made configurable after the runs reported in the paper. The scripts were compile-checked afterwards but the full pipeline was not re-run end to end from a clean clone; please open an issue if a step fails.
