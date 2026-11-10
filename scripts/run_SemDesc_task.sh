#!/usr/bin/env bash
# Generate phase diagram descriptions for the SemDesc task.
#
# Usage: bash scripts/run_SemDesc_task.sh
set -e

# Optional: name of the conda environment to activate before running.
if [ -n "${CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

# Repository root, resolved from the location of this script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Input task files.
INPUT_LIST=()

# Directory where the prediction JSON files are written.
OUTPUT_DIR=""

# Directory holding the phase diagram images referenced by the img_path field.
IMG_BASE_DIR="${PROJECT_ROOT}/dataset/MatPhaseBench-SemDesc/phase_images"

# Model backend: local | dashscope | jiekou | zhipu | tencent. Required.
BACKEND=""

# ---- Backend options: fill in the block matching BACKEND ----

# Local OpenAI-compatible endpoint.
LOCAL_MODEL=""
LOCAL_BASE_URL=""
LOCAL_API_KEY=""

# DashScope (Qwen).
DASHSCOPE_MODEL=""
DASHSCOPE_BASE_URL=""
DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"

# Jiekou aggregator.
JIEKOU_MODEL=""
JIEKOU_API_KEY="${JIEKOU_API_KEY:-}"

# Zhipu (GLM).
ZHIPU_MODEL=""
ZHIPU_API_KEY="${ZHIPU_API_KEY:-}"

# Tencent TokenHub.
TENCENT_MODEL=""
TENCENT_BASE_URL=""
TENCENT_API_KEY="${TENCENT_API_KEY:-}"


# Number of concurrent requests.
MAX_WORKERS=""

# Sampling temperature.
TEMPERATURE=""

# Nucleus sampling probability.
TOP_P=""

# Re-run samples whose previous output could not be parsed.
REPROCESS_ABNORMAL="false"

# Description mode: specified_dimension | unlimited_description |
# one_shot_description | three_shot_description.
DESCRIPTION_MODE="unlimited_description"

REPROCESS_ABNORMAL_ARGS=()
if [[ "${REPROCESS_ABNORMAL}" == "true" ]]; then
  REPROCESS_ABNORMAL_ARGS=(--reprocess-abnormal)
fi

if [[ -z "${BACKEND}" ]]; then
  echo "BACKEND is not set. Choose one of: local, dashscope, jiekou, zhipu, tencent." >&2
  exit 1
fi

# Arguments shared by every backend.
COMMON_ARGS=(
  --input "${INPUT_LIST[@]}"                  # input task files
  --output-dir "${OUTPUT_DIR}"                # prediction output directory
  --img-base-dir "${IMG_BASE_DIR}"            # image directory
  --backend "${BACKEND}"                      # model backend
  --temperature "${TEMPERATURE}"              # sampling temperature
  --top-p "${TOP_P}"                          # nucleus sampling probability
  --save-interval 1                           # flush predictions every N samples
  --max-workers "${MAX_WORKERS}"              # concurrent requests
  --description-mode "${DESCRIPTION_MODE}"    # description mode
  "${REPROCESS_ABNORMAL_ARGS[@]}"             # optional reprocess flag
)

if [[ "${BACKEND}" == "local" ]]; then
  # Local OpenAI-compatible endpoint; requires LOCAL_MODEL, LOCAL_BASE_URL, LOCAL_API_KEY.
  BACKEND_ARGS=(
    --local-model "${LOCAL_MODEL}"            # served model name
    --local-base-url "${LOCAL_BASE_URL}"      # endpoint base URL
    --local-api-key "${LOCAL_API_KEY}"        # endpoint API key
  )
elif [[ "${BACKEND}" == "dashscope" ]]; then
  # DashScope (Qwen); requires DASHSCOPE_MODEL, DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY.
  BACKEND_ARGS=(
    --model "${DASHSCOPE_MODEL}"              # model name
    --dashscope-base-url "${DASHSCOPE_BASE_URL}"  # endpoint base URL
    --dashscope-api-key "${DASHSCOPE_API_KEY}"    # API key from the environment
  )
elif [[ "${BACKEND}" == "jiekou" ]]; then
  # Jiekou aggregator; requires JIEKOU_MODEL and JIEKOU_API_KEY.
  BACKEND_ARGS=(
    --model "${JIEKOU_MODEL}"                 # model name
    --jiekou-api-key "${JIEKOU_API_KEY}"      # API key from the environment
  )
elif [[ "${BACKEND}" == "zhipu" ]]; then
  # Zhipu (GLM); requires ZHIPU_MODEL and ZHIPU_API_KEY.
  BACKEND_ARGS=(
    --model "${ZHIPU_MODEL}"                  # model name
    --zhipu-api-key "${ZHIPU_API_KEY}"        # API key from the environment
  )
elif [[ "${BACKEND}" == "tencent" ]]; then
  # Tencent TokenHub; requires TENCENT_MODEL, TENCENT_BASE_URL, TENCENT_API_KEY.
  BACKEND_ARGS=(
    --model "${TENCENT_MODEL}"                # model name
    --tencent-base-url "${TENCENT_BASE_URL}"  # endpoint base URL
    --tencent-api-key "${TENCENT_API_KEY}"    # API key from the environment
  )
else
  echo "Unsupported BACKEND: ${BACKEND}. Use local, dashscope, jiekou, zhipu, or tencent." >&2
  exit 1
fi

python "${PROJECT_ROOT}/src/phase_diagram_understanding_task/obtain_SemDesc_task_result.py" \
  "${COMMON_ARGS[@]}" \
  "${BACKEND_ARGS[@]}"
