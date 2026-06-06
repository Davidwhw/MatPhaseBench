#!/usr/bin/env bash
set -e

source your_python_environment

PROJECT_ROOT=""

INPUT="${PROJECT_ROOT}/dataset/MatPhaseBench.json"
OUTPUT_DIR="${PROJECT_ROOT}/your_prediction_result_storage_path/"
IMG_BASE_DIR="${PROJECT_ROOT}/dataset/"


BACKEND="dashscope"
DASHSCOPE_MODEL=""
DASHSCOPE_API_KEY=""

MAX_TOKENS="32768"
MAX_WORKERS="10"
REPROCESS_ABNORMAL="true"

REPROCESS_ABNORMAL_ARGS=()
if [[ "${REPROCESS_ABNORMAL}" == "true" ]]; then
  REPROCESS_ABNORMAL_ARGS=(--reprocess-abnormal)
fi

COMMON_ARGS=(
  --input "${INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --img-base-dir "${IMG_BASE_DIR}"
  --backend "${BACKEND}"
  --max-tokens "${MAX_TOKENS}"
  --save-interval 1
  --max-workers "${MAX_WORKERS}"
  "${REPROCESS_ABNORMAL_ARGS[@]}"
)

if [[ "${BACKEND}" == "dashscope" ]]; then
  BACKEND_ARGS=(
    --model "${DASHSCOPE_MODEL}"
    --dashscope-api-key "${DASHSCOPE_API_KEY}"
  )
elif [[ "${BACKEND}" == "shiyun" ]]; then
  BACKEND_ARGS=(
    --model "${SHIYUN_MODEL}"
    --shiyun-api-key "${SHIYUN_API_KEY}"
  )
elif [[ "${BACKEND}" == "zhipu" ]]; then
  BACKEND_ARGS=(
    --model "${ZHIPU_MODEL}"
    --zhipu-api-key "${ZHIPU_API_KEY}"
  )
else
  echo "Unsupported BACKEND: ${BACKEND}. Use dashscope, shiyun, or zhipu." >&2
  exit 1
fi

python "${PROJECT_ROOT}/src/obtain_MatPhaseBench_task_result.py" \
  "${COMMON_ARGS[@]}" \
  "${BACKEND_ARGS[@]}"
