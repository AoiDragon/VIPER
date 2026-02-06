#!/bin/bash
set -e 

# -----------------------------
# Defaults (can be overridden by args)
# -----------------------------
DATA_PATH="./results/video_inference/test_doubao-seedance-1-5-pro-251215"
OUTPUT_PATH="./results/vlm_judge/test_doubao-seedance-1-5-pro-251215"
FILE_NAME=""
FPS=1.0
MODEL_NAME="gpt-5-chat-2025-08-07"
PASS_K=1
MAX_WORKERS=8
SYSTEM_PROMPT_PATH="eval/prompt/system_prompt.txt"
DOMAIN_PROMPT_ROOT="eval/prompt/domain_prompt"

EVAL_ONLY=false
SCORE_ONLY=false
RESUME=false

# -----------------------------
# Resolve script paths
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/gpt-4o.py"
SCORE_SCRIPT="${SCRIPT_DIR}/score.py"
SCORE_SINGLE_SCRIPT="${SCRIPT_DIR}/score_single.py"

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --data_path PATH           Input data path (file or directory)
  --output_path PATH         Output directory path
  --file_name NAME           Evaluate only this file (only valid if data_path is a directory)
  --fps N                    Frame sampling rate (default: 1.0)
  --model_name NAME          Judge model name (default: ${MODEL_NAME})
  --pass_k K                 pass@k value k (default: ${PASS_K})
  --max_workers N            Worker count / concurrency (default: ${MAX_WORKERS})
  --system_prompt PATH       System prompt file path
  --domain_prompt_root PATH  Domain prompt directory path
  --eval_only                Run evaluation only (skip scoring)
  --score_only               Run scoring only (skip evaluation)
  --resume                   Resume mode: skip already-successful items, re-run missing/failed ones
  -h, --help                 Show this help

Examples:
  ./eval_and_score_multi.sh --data_path results/test_wan2.6-i2v --fps 1.0 --pass_k 1 --max_workers 8
  ./eval_and_score_multi.sh --data_path results/test_wan2.6-i2v --file_name specific_file.json
  ./eval_and_score_multi.sh --data_path results/test_wan2.6-i2v --eval_only
  ./eval_and_score_multi.sh --data_path results/test_wan2.6-i2v --score_only
EOF
  exit 1
}

# -----------------------------
# Parse
# -----------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path) DATA_PATH="$2"; shift 2 ;;
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --file_name) FILE_NAME="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --pass_k) PASS_K="$2"; shift 2 ;;
    --max_workers) MAX_WORKERS="$2"; shift 2 ;;
    --system_prompt) SYSTEM_PROMPT_PATH="$2"; shift 2 ;;
    --domain_prompt_root) DOMAIN_PROMPT_ROOT="$2"; shift 2 ;;
    --eval_only) EVAL_ONLY=true; shift ;;
    --score_only) SCORE_ONLY=true; shift ;;
    --resume) RESUME=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# -----------------------------
# Validate
# -----------------------------
[[ -f "$EVAL_SCRIPT" ]] || { echo "Error: Evaluation script not found: $EVAL_SCRIPT"; exit 1; }
[[ -f "$SCORE_SCRIPT" ]] || { echo "Error: Score script not found: $SCORE_SCRIPT"; exit 1; }
[[ -e "$DATA_PATH" ]] || { echo "Error: Data path not found: $DATA_PATH"; exit 1; }

if [[ -z "$OUTPUT_PATH" ]]; then
  if [[ -f "$DATA_PATH" ]]; then
    BASE_DIR="$(dirname "$DATA_PATH")"
  else
    BASE_DIR="$DATA_PATH"
  fi
  OUTPUT_PATH="$(echo "$BASE_DIR" | sed 's/inference/evaluation/')"
fi

# -----------------------------
# Log run info
# -----------------------------
echo "========================================"
echo "Video Reasoning Bench - Evaluation Pipeline"
echo "========================================"
echo "Data path:         $DATA_PATH"
echo "File name:         ${FILE_NAME:-'(all files)'}"
echo "Output path:       $OUTPUT_PATH"
echo "Judge Model:       $MODEL_NAME"
echo "FPS:               $FPS"
echo "Pass@K:            $PASS_K"
echo "Max Workers:       $MAX_WORKERS"
echo "Eval only:         $EVAL_ONLY"
echo "Score only:        $SCORE_ONLY"
echo "Resume mode:       $RESUME"
echo "========================================"
echo

# Step 1: Evaluation (unless --score_only)
if [[ "$SCORE_ONLY" = false ]]; then
  echo "========================================="
  echo "Step 1: Evaluation"
  echo "========================================="

  CMD=(python "$EVAL_SCRIPT"
    --system_prompt_path "$SYSTEM_PROMPT_PATH"
    --domain_prompt_root "$DOMAIN_PROMPT_ROOT"
    --data_path "$DATA_PATH"
    --output_path "$OUTPUT_PATH"
    --fps "$FPS"
    --model_name "$MODEL_NAME"
    --max_workers "$MAX_WORKERS"
    --k "$PASS_K"
  )

  [[ -n "$FILE_NAME" ]] && CMD+=(--file_name "$FILE_NAME")
  [[ "$RESUME" = true ]] && CMD+=(--resume)

  "${CMD[@]}"

  echo
  echo "✓ Evaluation completed successfully"
  echo
fi

# Step 2: Scoring (unless --eval_only)
if [[ "$EVAL_ONLY" = false ]]; then
  echo "========================================="
  echo "Step 2: Scoring"
  echo "========================================="

  if [[ -n "$FILE_NAME" ]]; then
    # Pick the newest matching eval result for this file/fps/k (ignore *_scores.json)
    EVAL_RESULT_FILE="$(
      ls -t "$OUTPUT_PATH/${FILE_NAME}_fps@${FPS}_pass@${PASS_K}"*.json 2>/dev/null \
        | grep -v "_scores.json" \
        | head -1
    )"

    if [[ -n "$EVAL_RESULT_FILE" ]]; then
      echo "Scoring single file: $(basename "$EVAL_RESULT_FILE")"
      python "$SCORE_SINGLE_SCRIPT" --input_file "$EVAL_RESULT_FILE" --fps "$FPS" --k "$PASS_K"
    else
      echo "Warning: No evaluation result file found matching ${FILE_NAME}_fps@${FPS}_pass@${PASS_K}*.json"
    fi
  else
    python "$SCORE_SCRIPT" --input_path "$OUTPUT_PATH" --output_path "$OUTPUT_PATH" --fps "$FPS" --k "$PASS_K"
  fi

  echo
  echo "✓ Scoring completed successfully"
  echo
fi
