#!/usr/bin/env bash
# Main experiments (80B): COEX on the 12 BRIGHT domains and the 7 BRIGHT-Pro domains, then evaluation.
# BRIGHT domains run first: each BRIGHT-Pro domain reuses the cached probes of its BRIGHT domain (same query ids).
# Usage: bash scripts/run_main.sh [URL, default http://localhost:8000/v1]
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${1:-http://localhost:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
OUT="${OUT:-results/coex}"
BRIGHT="biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems"
BRIGHT_PRO="bp_biology bp_earth_science bp_economics bp_psychology bp_robotics bp_stackoverflow bp_sustainable_living"
for T in $BRIGHT $BRIGHT_PRO; do
  python -m coex.run --task "$T" --url "$URL" --model "$MODEL" --out_dir "$OUT"
done
python eval/eval_bright.py --results "$OUT"
python eval/eval_bright_pro.py --results "$OUT" --bright_pro_repo external/Bright-Pro
