import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "new_stats.py")).read(); sys.argv = [sys.argv[0], "none"]; exec(src.split("part = sys.argv[1]")[0])
te = load("pool_all_cnn_D4")[0][1]
put("lc_pool_all_cnn_D4_vs_lc_mix36", boot(P("pool_all_cnn_D4"), P("lc_mix36"), te))
put("lc_lc_mix72_vs_lc_ds1_18_matched", boot(P("lc_mix72"), P("lc_ds1_18_matched"), te))
