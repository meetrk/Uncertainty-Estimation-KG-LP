#!/bin/bash
# run_remaining_models.sh - CORRECTED PATH PARSING

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/eval_lp_uncertainty_20runs.py"
EXPERIMENT_DIR="new_models/experiment_1"
EXISTING_RESULTS_DIR="results_lp_uncertainty"  # Your existing folder
NEW_RESULTS_ROOT="baseline_results_remaining"
NUM_RUNS=20
DATASET="WN18RR"

# === FIXED: CORRECT PATH EXTRACTION ===
echo "Scanning EXISTING results: $EXISTING_RESULTS_DIR/$DATASET"
EXISTING_RESULTS=()
while IFS= read -r -d '' file; do
    # Full path: .../results_lp_uncertainty/WN18RR/model_name/baseline/baseline/metrics.json
    # Extract model_name (2 levels up from metrics.json)
    MODEL_PATH=$(dirname "$(dirname "$file")")
    MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")  # model_name
    EXISTING_RESULTS+=("$MODEL_NAME")
done < <(find "$EXISTING_RESULTS_DIR/$DATASET" -path "*/baseline/baseline/metrics.json" -print0 2>/dev/null || true)

echo "✅ Already found (${#EXISTING_RESULTS[@]} models):"
printf '  %s\n' "${EXISTING_RESULTS[@]}" | sort -u
echo

# === FIND REMAINING ===
echo "🔍 Finding remaining WN18RR models in $EXPERIMENT_DIR..."
REMAINING_MODELS=()
while IFS= read -r -d '' file; do
    MODEL_NAME=$(basename "$file")
    if [[ ! " ${EXISTING_RESULTS[*]} " =~ " ${MODEL_NAME} " ]]; then
        REMAINING_MODELS+=("$MODEL_NAME")
    fi
done < <(find "$EXPERIMENT_DIR" -name "WN18RR_checkpoint*" -print0)

echo "🚀 Remaining to process (${#REMAINING_MODELS[@]}):"
printf '  %s\n' "${REMAINING_MODELS[@]}"
echo

if [ ${#REMAINING_MODELS[@]} -eq 0 ]; then
    echo "✅ NOTHING TO DO - ALL PROCESSED!"
    exit 0
fi

# === RUN REMAINING ===
echo "⚡ Starting baseline evaluation..."
for MODEL_NAME in "${REMAINING_MODELS[@]}"; do
    echo "=================================================="
    echo "📈 BASELINE: $MODEL_NAME"
    echo "=================================================="
    
    python "$PYTHON_SCRIPT" \
        --dataset "$DATASET" \
        --model "$MODEL_NAME" \
        --experiment-dir "$EXPERIMENT_DIR" \
        --results-root "$NEW_RESULTS_ROOT" \
        --num-runs "$NUM_RUNS" \
        --baseline-uncertainty \
        --mc-samples 5
    
    echo "✓ ✅ $MODEL_NAME → $NEW_RESULTS_ROOT/$DATASET/$MODEL_NAME/"
done

echo "=================================================="
echo "🎉 ALL REMAINING COMPLETE!"
echo "📁 Location: $NEW_RESULTS_ROOT/$DATASET/"
ls -la "$NEW_RESULTS_ROOT/$DATASET/" | head -10
echo "=================================================="
