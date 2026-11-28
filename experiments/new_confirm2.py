"""Secondary comparisons of the confirmatory test (European ST-T Database), from the saved predictions. Fast bootstrap."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fastboot as FB
OUTF = (WORK + "/new_stats.json"); OUT = json.load(open(OUTF))
E = np.load((WORK + "/confirm_pred_edb.npz")); y = E["y"]; S = E["subjects"]
M = {n: {s: E["%s_%d" % (k, s)] for s in range(10)} for n, k in (("hyb", "h"), ("learned", "l4"), ("learned32", "l32"), ("rr", "rr"))}
for kind in ("macro", "abn", "clsN", "clsS", "clsV"):
    for c in ("learned32", "learned", "rr"):
        k = "confirm_edb_hyb14_vs_%s_%s" % (c, kind)
        if k == "confirm_edb_hyb14_vs_learned32_macro": continue          # primary: exact bootstrap, already stored
        OUT[k] = FB.boot(M["hyb"], M[c], y, S, kind, 2000); json.dump(OUT, open(OUTF, "w"), indent=1); v = OUT[k]
        print("%-40s a %.3f b %.3f d %+.3f [%+.3f,%+.3f] p %.3f" % (k, v["a"], v["b"], v["diff"], *v["ci95"], v["p"]), flush=True)
