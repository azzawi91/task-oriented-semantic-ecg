#!/usr/bin/env bash
# One-command launcher for the revision experiments (runs in the background,
# keeps the Mac awake with caffeinate, resumable: just run it again if interrupted).
#
#   bash run.sh                      # uses ../physionet.org - Databases/files/mitdb/1.0.0
#   bash run.sh /path/to/mitdb/1.0.0 # explicit MIT-BIH folder
#   bash run.sh "" --skip-ptbxl      # skip the PTB-XL download/training stage
#
# Watch progress:   tail -f run_log.txt      Stop:   kill $(cat run.pid)
cd "$(dirname "$0")"
if [ -f "$HOME/torch-env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/torch-env/bin/activate"
fi
if ! python3 -c "import torch, sklearn, scipy, numpy" 2>/dev/null; then
  echo "Installing missing packages into the active Python ..."
  python3 -m pip install --quiet torch numpy scipy scikit-learn
fi
MITDB="${1:-../physionet.org - Databases/files/mitdb/1.0.0}"
shift 2>/dev/null
if [ ! -f "$MITDB/100.dat" ]; then
  echo "MIT-BIH not found at: $MITDB   (expected 100.dat there)"; exit 1
fi
echo "Python: $(python3 -c 'import sys,torch; print(sys.version.split()[0], "torch", torch.__version__)')"
nohup caffeinate -i python3 run_all.py --mitdb "$MITDB" "$@" > run_stdout.txt 2>&1 &
echo $! > run.pid
echo "Started in the background (PID $(cat run.pid)). Progress: tail -f run_log.txt"
