# Revision experiments (TGCN manuscript)

Self-contained code for every experiment reported in the manuscript
*Task-Oriented Semantic Coding for Energy-Efficient and Post-Quantum-Secure Wearable ECG Telemetry* and the
post-hoc analysis that regenerates every table and figure. It depends only on
`numpy`, `scipy`, `scikit-learn`, `torch`, and `matplotlib`; `wfdb_lite.py` reads
MIT-BIH (format 212) and PTB-XL (format 16) records without the `wfdb` package.

## Run everything

```bash
cd revision
bash run.sh /path/to/mitdb/1.0.0                     # MIT-BIH folder that contains 100.dat ... 234.dat
# optional flags are passed through to run_all.py:
bash run.sh /path/to/mitdb/1.0.0 --skip-ptbxl        # skip the PTB-XL stage
bash run.sh /path/to/mitdb/1.0.0 --ptbxl-root /path/to/ptb-xl/1.0.3   # use an existing PTB-XL copy
```

`run.sh` starts `run_all.py` in the background (on macOS it keeps the machine
awake with `caffeinate`), logs to `run_log.txt` / `status.json`, and is
resumable: finished (experiment, seed) pairs are skipped, so the same command can
be re-run after an interruption. Results land in
`results_revision/<experiment>/seed<k>.{json,npz,pt}`.

| Stage | Experiment folder(s) | Seeds |
|---|---|---|
| 1 | window caches at 125 / 250 / 360 Hz, two-lead and MLII-only | – |
| 2 | `main` – published configuration, de Chazal DS1/DS2 | 10 |
| 3 | `recon_convae`, `recon_jscc`, `recon_sameenc_pe` – reconstruction codecs (60 / 60 / 30 epochs, cosine LR, validation-selected); `main_pe` – task model with the same positional encodings | 3 / 10 |
| 3b | `latent4`, `latent8` – 4- and 8-coefficient codecs | 10 / 5 |
| 4 | `abl_cnn_only`, `abl_tf1`, `abl_no_reg`, `abl_no_rate`, `abl_no_ee`, `abl_task_only` | 10 |
| 5 | `qat4`, `qat3` (quantization-aware), `chan_erase20` (erasure-aware) | 10 |
| 6 | `lead_mlii` – single-lead input | 10 |
| 7 | `fs250`, `fs360` – sampling-rate sweep | 10 |
| 8 | `random_split` – record-disjoint 60/20/20 | 3 |
| 9 | `ptbxl_full` – all PTB-XL records with a diagnostic superclass, official folds 1–8 / 9 / 10 | 5 |

Expect a few hours on a laptop CPU for stages 1–8 and roughly one more hour for
stage 9 (plus the 1.7 GB PTB-XL download unless `--ptbxl-root` is given).

## Regenerate the tables and figures

```bash
python analyze.py results_revision caches paper_out
```

writes `paper_out/tables/*.tex`, `paper_out/figs/*.pdf`, `paper_out/numbers.tex`
(every number quoted in the manuscript, as LaTeX macros) and
`paper_out/analysis.json`. The post-training quantization, coefficient-subset,
channel, energy, and post-quantum analyses in `analyze.py` are NumPy-only and use
the exported latents, decoder weights, and probabilities from the training runs.
Copy `tables/`, `figs/` and `numbers.tex` into `../manuscript/` and run
`bash build.sh` there to rebuild the PDF.

## Files

* `run_all.py` – orchestrator (resumable, logs, summary)
* `run.sh` – background launcher
* `rev_data.py`, `wfdb_lite.py` – MIT-BIH windowing and labels (AAMI EC57 window classes, rhythm annotations)
* `rev_model.py` – encoder/decoder with the ablation switches (identical to the manuscript's model by default) and an analytic MAC count that includes attention
* `rev_train.py` – trains one configuration/seed and exports latents, decoder weights, probabilities, and metrics
* `rev_recon.py` – reconstruction codecs (convolutional autoencoder, deep JSCC, same-encoder + positional encodings)
* `rev_ptbxl.py` – PTB-XL download and cache builder
* `rev_splits.py`, `rev_metrics.py` – protocols and metrics
* `rev_np.py` – NumPy re-implementation of the decoder used by the post-hoc analyses
* `analyze.py` – everything post hoc: tables, figures, `numbers.tex`
