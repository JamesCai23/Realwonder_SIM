#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export REALWONDER_INPAINT_DTYPE="${REALWONDER_INPAINT_DTYPE:-fp16}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_PATH="${1:-$ROOT_DIR/scripts/vlm_simulation.csv}"

PY_REALWONDER="${PY_REALWONDER:-/home/lff/miniconda3/envs/realwonder/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${2:-}" == "--dry_run" || "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

pick_gpus() {
  local n="${1:-1}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return 0
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr \
    | head -n "$n" \
    | awk -F',' '{gsub(/ /, "", $1); print $1}' \
    | paste -sd',' -
}

CUDA_REALWONDER="${CUDA_REALWONDER:-$(pick_gpus 2)}"

# Device placement inside this process-visible GPU list.
REALWONDER_DEVICE="${REALWONDER_DEVICE:-cuda:0}"
REALWONDER_SAM3_DEVICE="${REALWONDER_SAM3_DEVICE:-cuda:0}"
REALWONDER_INPAINT_DEVICE="${REALWONDER_INPAINT_DEVICE:-cuda:1}"
REALWONDER_NOISE_DEVICE="${REALWONDER_NOISE_DEVICE:-cuda:0}"
REALWONDER_INPAINT_OFFLOAD="${REALWONDER_INPAINT_OFFLOAD:-none}"
REALWONDER_PRELOAD_INPAINT="${REALWONDER_PRELOAD_INPAINT:-1}"
REALWONDER_RELEASE_INPAINT_BEFORE_NOISE="${REALWONDER_RELEASE_INPAINT_BEFORE_NOISE:-0}"

IFS=',' read -r -a _RW_GPU_ARR <<< "$CUDA_REALWONDER"
if [[ -z "$REALWONDER_INPAINT_DEVICE" ]]; then
  if [[ "${#_RW_GPU_ARR[@]}" -ge 2 ]]; then
    REALWONDER_INPAINT_DEVICE="cuda:1"
  else
    REALWONDER_INPAINT_DEVICE="cuda:0"
  fi
fi

if [[ -z "$REALWONDER_NOISE_DEVICE" ]]; then
  if [[ "${#_RW_GPU_ARR[@]}" -ge 2 ]]; then
    if [[ "$REALWONDER_INPAINT_DEVICE" == "cuda:0" ]]; then
      REALWONDER_NOISE_DEVICE="cuda:1"
    else
      REALWONDER_NOISE_DEVICE="cuda:0"
    fi
  else
    REALWONDER_NOISE_DEVICE="cuda:0"
  fi
fi

validate_visible_cuda_slot() {
  local var_name="$1"
  local value="$2"
  local fallback="$3"

  if [[ "$value" =~ ^cuda:([0-9]+)$ ]]; then
    local idx="${BASH_REMATCH[1]}"
    local visible_count="${#_RW_GPU_ARR[@]}"
    if [[ "$idx" -ge "$visible_count" ]]; then
      echo "WARN: $var_name=$value is outside visible CUDA slots (0..$((visible_count-1))). Fallback to $fallback" >&2
      printf -v "$var_name" '%s' "$fallback"
    fi
  fi
}

validate_visible_cuda_slot REALWONDER_DEVICE "$REALWONDER_DEVICE" "cuda:0"
validate_visible_cuda_slot REALWONDER_SAM3_DEVICE "$REALWONDER_SAM3_DEVICE" "$REALWONDER_DEVICE"
validate_visible_cuda_slot REALWONDER_INPAINT_DEVICE "$REALWONDER_INPAINT_DEVICE" "$REALWONDER_DEVICE"
validate_visible_cuda_slot REALWONDER_NOISE_DEVICE "$REALWONDER_NOISE_DEVICE" "$REALWONDER_DEVICE"

if [[ "$REALWONDER_NOISE_DEVICE" == "$REALWONDER_INPAINT_DEVICE" ]]; then
  echo "WARN: noise and inpaint share the same device ($REALWONDER_NOISE_DEVICE)." >&2
  echo "      Keep REALWONDER_RELEASE_INPAINT_BEFORE_NOISE=0 to preserve cache reuse." >&2
fi

if [[ "${#_RW_GPU_ARR[@]}" -lt 2 && "$REALWONDER_INPAINT_OFFLOAD" == "none" ]]; then
  REALWONDER_INPAINT_OFFLOAD="sequential"
fi

RW_OUTPUT_ROOT="${RW_OUTPUT_ROOT:-/home/lff/bigdata1/cym/realwonder_vlm_simdata}"
RW_CASE_ROOT="${RW_CASE_ROOT:-$RW_OUTPUT_ROOT/cases}"
RW_RESULT_ROOT="${RW_RESULT_ROOT:-$RW_OUTPUT_ROOT/result}"
QA_OUTPUT_CSV="${QA_OUTPUT_CSV:-$RW_RESULT_ROOT/quantiphy_synthetic_dataset.csv}"

if [[ -z "${VLLM_BASE_URL:-}" ]]; then
  if curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1; then
    VLLM_BASE_URL="http://localhost:8000/v1"
  elif curl -sf "http://localhost:8002/v1/models" >/dev/null 2>&1; then
    VLLM_BASE_URL="http://localhost:8002/v1"
  else
    VLLM_BASE_URL="http://localhost:8000/v1"
  fi
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: CSV not found: $CSV_PATH" >&2
  exit 1
fi
if [[ ! -x "$PY_REALWONDER" ]]; then
  echo "ERROR: realwonder python not found/executable: $PY_REALWONDER" >&2
  exit 1
fi

if [[ "${SKIP_ENV_CHECK:-0}" != "1" ]]; then
  "$PY_REALWONDER" -c "import yaml, torch, omegaconf, openai; print('realwonder env OK')" >/dev/null
fi

echo "== RW SimOnly batch =="
echo "ROOT_DIR=$ROOT_DIR"
echo "CSV_PATH=$CSV_PATH"
echo "PY_REALWONDER=$PY_REALWONDER"
echo "CUDA_REALWONDER=$CUDA_REALWONDER"
echo "REALWONDER_DEVICE=$REALWONDER_DEVICE"
echo "REALWONDER_SAM3_DEVICE=$REALWONDER_SAM3_DEVICE"
echo "REALWONDER_INPAINT_DEVICE=$REALWONDER_INPAINT_DEVICE"
echo "REALWONDER_NOISE_DEVICE=$REALWONDER_NOISE_DEVICE"
echo "REALWONDER_INPAINT_OFFLOAD=$REALWONDER_INPAINT_OFFLOAD"
echo "REALWONDER_PRELOAD_INPAINT=$REALWONDER_PRELOAD_INPAINT"
echo "REALWONDER_RELEASE_INPAINT_BEFORE_NOISE=$REALWONDER_RELEASE_INPAINT_BEFORE_NOISE"
echo "VLLM_BASE_URL=$VLLM_BASE_URL"
echo "QA_OUTPUT_CSV=$QA_OUTPUT_CSV"
echo "RW_CASE_ROOT=$RW_CASE_ROOT"
echo "RW_RESULT_ROOT=$RW_RESULT_ROOT"
echo "DRY_RUN=$DRY_RUN"

ARGS=(
  "$ROOT_DIR/batch_simulation_csv.py"
  --csv_path "$CSV_PATH"
  --stages qa
  --qa_output_csv "$QA_OUTPUT_CSV"
  --output_root "$RW_CASE_ROOT"
  --result_root "$RW_RESULT_ROOT"
)
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry_run)
fi

# Retry logic for CUDA OOM
MAX_RETRIES=${MAX_RETRIES:-10}
RETRY_COUNT=0
EXIT_CODE=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_REALWONDER" \
  REALWONDER_DEVICE="$REALWONDER_DEVICE" \
  REALWONDER_SAM3_DEVICE="$REALWONDER_SAM3_DEVICE" \
  REALWONDER_INPAINT_DEVICE="$REALWONDER_INPAINT_DEVICE" \
  REALWONDER_NOISE_DEVICE="$REALWONDER_NOISE_DEVICE" \
  REALWONDER_INPAINT_OFFLOAD="$REALWONDER_INPAINT_OFFLOAD" \
  REALWONDER_PRELOAD_INPAINT="$REALWONDER_PRELOAD_INPAINT" \
  REALWONDER_RELEASE_INPAINT_BEFORE_NOISE="$REALWONDER_RELEASE_INPAINT_BEFORE_NOISE" \
  VLLM_BASE_URL="$VLLM_BASE_URL" \
  "$PY_REALWONDER" "${ARGS[@]}"
  EXIT_CODE=$?
  set -e
  
  if [ $EXIT_CODE -eq 0 ]; then
    break
  fi

  echo "Process exited with code $EXIT_CODE. Retry $((RETRY_COUNT+1))/$MAX_RETRIES..." >&2
  RETRY_COUNT=$((RETRY_COUNT+1))
  sleep 2
done

if [ $EXIT_CODE -ne 0 ]; then
  echo "Max retries reached. Exiting with code $EXIT_CODE." >&2
  exit $EXIT_CODE
fi

echo "== Done =="
