#!/bin/bash
#SBATCH --account=rrg-glatard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=__JOBNAME__
#SBATCH --output=__WORKDIR__/extract_%j.out

set -euo pipefail

MODEL_SLUG="__MODEL_SLUG__"
DATASET="__DATASET__"
WORKDIR="__WORKDIR__"                     
INPUT_DIR="__INPUT_DIR__"                
WEIGHTS_DIR="$WORKDIR/weights"
EXTRACT_PY="$WORKDIR/extract.py"
OUT_CSV="$WORKDIR/${DATASET}.csv"
DONE_FLAG="$WORKDIR/${DATASET}.done"

echo "[$(date)] extraction start: model=$MODEL_SLUG dataset=$DATASET"
echo "  input_dir=$INPUT_DIR"
echo "  weights_dir=$WEIGHTS_DIR"

module load StdEnv/2023 python/3.11 2>/dev/null || true
source "$WORKDIR/venv/bin/activate" 2>/dev/null || true

# The contract: extract.py exposes  extract(input_dir, output_csv, weights_dir)
python - "$INPUT_DIR" "$OUT_CSV" "$WEIGHTS_DIR" "$EXTRACT_PY" <<'PYEOT'
import sys, importlib.util
input_dir, output_csv, weights_dir, extract_py = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("submission_extract", extract_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if not hasattr(mod, "extract"):
    raise SystemExit("submission extract.py must define extract(input_dir, output_csv, weights_dir)")
mod.extract(input_dir, output_csv, weights_dir)
PYEOT

# sanity: output must exist and be non-trivial
if [ ! -s "$OUT_CSV" ]; then
    echo "ERROR: extract.py produced no output CSV"
    exit 1
fi

# sentinel written ONLY on success
echo "$(date)" > "$DONE_FLAG"
echo "[$(date)] extraction done: wrote $OUT_CSV and $DONE_FLAG"
