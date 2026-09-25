"""Canonical de Chazal et al. (2004) inter-patient partition for MIT-BIH.

DS1 trains, DS2 tests; the four paced records (102, 104, 107, 217) are excluded,
as in the standard AAMI inter-patient evaluation. A small, seeded subset of DS1
records is held out for validation (record-disjoint from the training records),
so DS2 stays a fixed, untouched test set across seeds.
"""
import numpy as np

DS1 = ["101","106","108","109","112","114","115","116","118","119","122","124",
       "201","203","205","207","208","209","215","220","223","230"]
DS2 = ["100","103","105","111","113","117","121","123","200","202","210","212",
       "213","214","219","221","222","228","231","232","233","234"]
PACED = ["102","104","107","217"]


def split(y, groups, seed=0, n_val_records=4):
    """Return (train_idx, val_idx, test_idx) over cached windows.
    test = DS2 (fixed); val = seeded n_val_records held-out DS1 records; train = rest of DS1."""
    rng = np.random.default_rng(seed)
    ds1 = [r for r in DS1]
    rng.shuffle(ds1)
    val_recs = set(ds1[:n_val_records])
    tr_recs = set(ds1[n_val_records:])
    tr = np.where(np.isin(groups, list(tr_recs)))[0]
    va = np.where(np.isin(groups, list(val_recs)))[0]
    te = np.where(np.isin(groups, DS2))[0]
    return tr, va, te
