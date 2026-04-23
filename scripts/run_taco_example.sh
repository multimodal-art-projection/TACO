#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# TACO + terminus-2 example runner.
#
# Minimal runnable template. See docs for parameter reference:
#   src/harbor/agents/terminus_2/README.md
# ---------------------------------------------------------------------------
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# ---- main agent model --------------------------------------------------
MODEL_NAME="openai/gpt-4o-mini"
API_BASE=""                       # e.g. "https://api.openai.com/v1"
API_KEY="EMPTY"
export OPENAI_API_KEY="$API_KEY"

# ---- compression LLM ---------------------------------------------------
COMPRESS_BASE_URL="<YOUR_COMPRESS_BASE_URL>"
COMPRESS_API_KEY="<YOUR_COMPRESS_API_KEY>"
COMPRESS_MODEL_NAME="<YOUR_COMPRESS_MODEL>"

# ---- run config --------------------------------------------------------
DATASET="terminal-bench@2.0"
NUM_TASKS=16
NUM_ITERATIONS=1
OUTPUT_DIR="results/taco_example"

MAX_INPUT_TOKENS=132000
MAX_OUTPUT_TOKENS=32768

mkdir -p "$OUTPUT_DIR"

for i in $(seq 1 "$NUM_ITERATIONS"); do
  echo "=========================================================="
  echo "Iteration $i / $NUM_ITERATIONS  --  dataset=$DATASET  n=$NUM_TASKS"
  echo "=========================================================="

  harbor run \
    -d "$DATASET" \
    -a terminus-2 \
    -m "$MODEL_NAME" \
    -n "$NUM_TASKS" \
    -o "$OUTPUT_DIR" \
    --timeout-multiplier 1.0 \
    --ak key="$API_KEY" \
    --ak api_base="$API_BASE" \
    --ak enable_compress=True \
    --ak compress_base_url="$COMPRESS_BASE_URL" \
    --ak compress_api_key="$COMPRESS_API_KEY" \
    --ak compress_model_name="$COMPRESS_MODEL_NAME" \
    --ak enable_self_evo=True \
    --ak freeze_rules=False \
    --ak disable_global_evo=False \
    --ak uncovered_threshold=3000 \
    --ak model_info="{\"max_input_tokens\": $MAX_INPUT_TOKENS, \"max_output_tokens\": $MAX_OUTPUT_TOKENS, \"input_cost_per_token\": 0.0, \"output_cost_per_token\": 0.0}"

  echo "Iteration $i complete."
done

echo "All done. Results written to: $OUTPUT_DIR"
