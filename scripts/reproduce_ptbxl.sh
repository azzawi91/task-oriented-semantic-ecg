#!/usr/bin/env bash
# Second-dataset (PTB-XL) experiment on CPU. No GPU needed.
# 1) download PTB-XL 100 Hz (~1.7 GB) 2) build cache 3) train full schedule.
set -e; cd "$(dirname "$0")"
ROOT="${1:-data/ptbxl}"
mkdir -p "$ROOT"
if [ ! -f "$ROOT/ptbxl_database.csv" ]; then
  wget -q -c -np -nH --cut-dirs=3 -P "$ROOT" "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv" "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
fi
# download 100 Hz records (records100/) -- resumable; re-run if interrupted
wget -q -c -np -nH --cut-dirs=3 -r -A "*.hea,*.dat" "https://physionet.org/files/ptb-xl/1.0.3/records100/" -P "$ROOT" || true
python3 data_ptbxl.py "$ROOT"
python3 train_ptbxl.py 6 100000 64
echo "Done -> results/ptbxl_result.json"
