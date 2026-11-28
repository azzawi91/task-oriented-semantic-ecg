# Confirmatory test of the hybrid code on a fourth database — analysis plan

Written 27 November 2026, before the European ST-T Database was downloaded or opened.

**Database.** European ST-T Database (PhysioNet `edb` 1.0.0): 90 two-channel ambulatory records of two hours, 250 Hz, cardiologist beat annotations. It has not been used anywhere in this study.

**Preparation (identical to the other databases).** Both channels as recorded, in file order; resampled to 125 Hz; 10-s non-overlapping windows; per-channel standardisation; window label VEB / SVEB / normal by the same rule; windows with no classifiable beat dropped. RR-interval features from the same detector on channel 0.

**Frozen models.** The ten three-database models with the 4-byte code (`pool_all_cnn_D4`, seeds 0-9) and the ten with the 32-byte code (`pool_all_cnn_D32`, seeds 0-9), as already trained. No retraining, no fine-tuning, no recalibration, no use of any window of this database for fitting or scaling.

**Codes compared.**
- Hybrid 14 B: 4 learned bytes + the ten RR-interval statistics, read by the class-balanced logistic regression fitted on each model's own training windows (exactly as in `new_hybrid.py`).
- Learned 32 B: trained decoder of the 32-byte model.
- Learned 4 B: trained decoder of the 4-byte model.
- RR 22 B: logistic regression on the 22 quantised hand-crafted coefficients, fitted on the same training windows.

**Primary hypothesis.** Hybrid 14 B has a higher three-class macro-averaged one-versus-rest AUROC than Learned 32 B.
Test: paired two-level bootstrap, 2000 replicates, resampling subjects and seeds. The hypothesis is confirmed if the two-sided 95% interval of the difference lies above zero.

**Secondary (reported whatever the outcome).** Hybrid vs Learned 4 B; Hybrid vs RR 22 B; the same three comparisons for abnormal-versus-normal; per-class AUROC.

**Resampling unit.** The database has 90 records from 79 subjects and does not publish the mapping. Records whose header gives the same age, sex and diagnoses are treated as one subject; otherwise each record is its own unit.

**Rules.** The analysis is run once. The result is reported whether or not the hypothesis is confirmed. No comparison, metric, feature set or exclusion is changed after the data are opened.
