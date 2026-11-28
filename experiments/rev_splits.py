"""Evaluation protocols (identical to the published code: dechazal.py and
run_mitbih_experiment.patient_split)."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import numpy as np

DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
       "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
       "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
PACED = ["102", "104", "107", "217"]


def dechazal_split(y, groups, seed=0, n_val_records=4):
    """DS2 = fixed test set; a seeded subset of DS1 records is held out for
    threshold calibration; the remaining DS1 records train."""
    rng = np.random.default_rng(seed)
    ds1 = list(DS1); rng.shuffle(ds1)
    val_recs, tr_recs = set(ds1[:n_val_records]), set(ds1[n_val_records:])
    tr = np.where(np.isin(groups, list(tr_recs)))[0]
    va = np.where(np.isin(groups, list(val_recs)))[0]
    te = np.where(np.isin(groups, DS2))[0]
    return tr, va, te


def record_disjoint_split(y, groups, seed, fracs=(0.6, 0.2, 0.2), min_per_class=2, tries=200):
    """Random record-disjoint 60/20/20 split over all 48 records (secondary protocol)."""
    rng = np.random.default_rng(seed)
    recs = np.array(sorted(set(groups)))
    classes = np.unique(y); last = None
    for _ in range(tries):
        rng.shuffle(recs)
        n = len(recs); a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        sets = (set(recs[:a]), set(recs[a:b]), set(recs[b:]))
        idx = [np.where(np.isin(groups, list(S)))[0] for S in sets]
        last = idx
        if all(len(ix) > 0 and all((y[ix] == k).sum() >= min_per_class for k in classes) for ix in idx):
            return idx[0], idx[1], idx[2]
    return last[0], last[1], last[2]


def fold_split(groups):
    """PTB-XL official stratified folds stored as strings '1'..'10' in `groups`."""
    f = groups.astype(int)
    return np.where(f <= 8)[0], np.where(f == 9)[0], np.where(f == 10)[0]


def patient_ids(groups, incart_root=(WORK + "/data/incartdb/1.0.0")):
    """Patient identifier per window. MIT-BIH records 201 and 202 are one subject; INCART records are mapped to
    patients from the '# patient N' line of each header; paced MIT-BIH records return None."""
    import os, re
    cache = {}
    def one(gname):
        if gname in cache: return cache[gname]
        if gname.startswith("incart_"):
            rec = gname.split("_", 1)[1]; txt = open(os.path.join(incart_root, rec + ".hea"), errors="ignore").read()
            m = re.search(r"#\s*patient\s+(\d+)", txt); v = "incart:p%s" % (m.group(1) if m else rec)
        elif gname.startswith("svdb_"): v = "svdb:" + gname.split("_", 1)[1]
        elif gname in DS1 or gname in DS2: v = "mit:" + ("201" if gname == "202" else gname)
        else: v = None
        cache[gname] = v; return v
    return [one(str(x)) for x in groups]
