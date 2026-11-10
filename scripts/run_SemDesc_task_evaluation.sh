#!/usr/bin/env bash
# Evaluate SemDesc predictions with traditional metrics or LLM-as-Judge.
#
# Usage: bash scripts/run_SemDesc_task_evaluation.sh
set -e

# Optional: name of the conda environment to activate before running.
if [ -n "${CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

# GPUs exposed to BERTScore; set to e.g. "0" or "0,1" to pin devices.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Repository root, resolved from the location of this script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Metric mode: traditional | llm_as_judge.
METRIC_MODE="llm_as_judge"

# Model name used to locate the prediction files. Required.
TEST_MODEL=""

# Directory holding the prediction JSON files.
PREDICTION_DIR=""

# Prediction files to evaluate; one record per model and split.
PREDICTION_FILES=()

# Optional image directory passed to the evaluator (empty: images referenced by the predictions).
IMAGE_DIR=()

# Output file holding the per-sample metrics records.
OUTPUT=""

# Re-score only the samples that failed in a previous run.
LLM_REPROCESS_FAILED=false

# Source file to resume failed samples from.
LLM_REPROCESS_FAILED_FROM="${OUTPUT}"

# Optional JSON listing the samples to evaluate (empty: evaluate every sample).
DIMENSION_SAMPLE_SELECTION=

# BERTScore backbone directory (see BERTScore_model/README.md).
BERTSCORE_XLNET_MODEL="${PROJECT_ROOT}/BERTScore_model/xlnet-large-cased"

# Number of transformer layers used for BERTScore.
BERTSCORE_XLNET_NUM_LAYERS=24

# Rescaling baseline TSV shipped with the backbone.
BERTSCORE_XLNET_BASELINE_PATH="${BERTSCORE_XLNET_MODEL}/xlnet-large-cased.tsv"

# Device used for BERTScore.
BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cuda:0}"

# Pass --bertscore-rescale-with-baseline; clear it to report raw BERTScore.
BERTSCORE_RESCALE_WITH_BASELINE_OPTION="--bertscore-rescale-with-baseline"

# Number of parallel workers for traditional metrics.
TRAD_WORKERS=4

# Dimensions reported in the metrics records.
DIMENSION_LABELS=(
  "materials_system"
  "diagram_type"
  "diagram_completeness"
  "phase_regions_boundaries"
  "invariant_reactions"
)

# LLM-as-Judge backend; leave empty to use the evaluator's default backend.
LLM_BACKEND=""

# LLM-as-Judge model name; required when METRIC_MODE=llm_as_judge.
LLM_MODEL=""

# API key of the judge model, read from the environment (never hard-coded).
LLM_API_KEY="${LLM_API_KEY:-}"

# Enable reasoning mode of the judge model; empty array keeps it disabled.
LLM_ENABLE_THINKING_OPTION=()

# Number of retries per failed judge request.
LLM_RETRY_COUNT=3

# Seconds to wait between retries.
LLM_RETRY_SLEEP_SECONDS=5

# Number of concurrent judge requests.
LLM_MAX_WORKERS=16

# Partial results file, used to resume an interrupted judge run.
LLM_PARTIAL_OUTPUT="${OUTPUT%.json}.partial.json"

if [[ -z "${TEST_MODEL}" ]]; then
  echo "TEST_MODEL is not set. Provide the model name used in the prediction file names." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

build_common_args() {
  COMMON_ARGS=(
    --metric-mode "${METRIC_MODE}"                              # metric mode
    --prediction "${PREDICTION_FILES[@]}"                       # prediction files
    --image-dir "${IMAGE_DIR}"                                  # optional image directory
    --output "${OUTPUT}"                                        # metrics output file
    --bertscore-xlnet-model "${BERTSCORE_XLNET_MODEL}"          # BERTScore backbone
    --bertscore-xlnet-num-layers "${BERTSCORE_XLNET_NUM_LAYERS}" # backbone layer count
    --bertscore-xlnet-baseline-path "${BERTSCORE_XLNET_BASELINE_PATH}" # rescaling baseline
    --bertscore-device "${BERTSCORE_DEVICE}"                    # BERTScore device
    "${BERTSCORE_RESCALE_WITH_BASELINE_OPTION}"                 # optional rescaling flag
    --trad-workers "${TRAD_WORKERS}"                            # traditional metric workers
    --dimension-labels "${DIMENSION_LABELS[@]}"                 # reported dimensions
    --skip-dimension-aggregation                                # keep per-dimension records
  )
  if [[ -n "${DIMENSION_SAMPLE_SELECTION}" ]]; then
    COMMON_ARGS+=(--dimension-sample-selection "${DIMENSION_SAMPLE_SELECTION}") # sample subset
  fi
}

build_llm_as_judge_args() {
  LLM_ARGS=()

  if [[ "${METRIC_MODE}" != "llm_as_judge" ]]; then
    return
  fi

  if [[ -z "${LLM_MODEL}" ]]; then
    echo "LLM_MODEL must be set when METRIC_MODE=llm_as_judge." >&2
    exit 1
  fi
  if [[ "${LLM_REPROCESS_FAILED}" == "true" && ! -f "${LLM_REPROCESS_FAILED_FROM}" ]]; then
    echo "LLM_REPROCESS_FAILED_FROM does not exist: ${LLM_REPROCESS_FAILED_FROM}" >&2
    exit 1
  fi

  LLM_ARGS=(
    --llm-model "${LLM_MODEL}"                                  # judge model name
    "${LLM_ENABLE_THINKING_OPTION[@]}"                          # optional reasoning flag
    --llm-retry-count "${LLM_RETRY_COUNT}"                      # retries per request
    --llm-retry-sleep-seconds "${LLM_RETRY_SLEEP_SECONDS}"      # delay between retries
    --llm-max-workers "${LLM_MAX_WORKERS}"                      # concurrent requests
    --llm-partial-output "${LLM_PARTIAL_OUTPUT}"                # partial results file
  )
  if [[ -n "${LLM_JUDGE_DETAIL_OUTPUT:-}" ]]; then
    LLM_ARGS+=(--llm-judge-detail-output "${LLM_JUDGE_DETAIL_OUTPUT}") # per-sample reasons
  fi
  if [[ -n "${LLM_BACKEND}" ]]; then
    LLM_ARGS+=(--llm-backend "${LLM_BACKEND}")                  # judge backend
  fi
  if [[ -n "${LLM_API_KEY}" ]]; then
    LLM_ARGS+=(--llm-api-key "${LLM_API_KEY}")                  # judge API key
  fi
  if [[ "${LLM_REPROCESS_FAILED}" == "true" ]]; then
    LLM_ARGS+=(--llm-reprocess-failed-from "${LLM_REPROCESS_FAILED_FROM}") # resume source
  fi
}

build_common_args
build_llm_as_judge_args

python "${PROJECT_ROOT}/src/task_evaluation/SemDesc_task_evaluation.py" \
  "${COMMON_ARGS[@]}" \
  "${LLM_ARGS[@]}"
