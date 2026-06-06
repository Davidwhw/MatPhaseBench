#!/usr/bin/env bash
set -e

source your_python_environment

PROJECT_ROOT=""

METRIC_MODE="traditional"
MODEL_NAME=""

PREDICTION="${PROJECT_ROOT}/your_prediction_result_storage_path/prediction_result.json"
GROUND_TRUTH="${PROJECT_ROOT}/dataset/MatPhaseBench.json"
OUTPUT="${PROJECT_ROOT}/your_prediction_result_storage_path/${METRIC_MODE}_dimensions_${MODEL_NAME}_metrics_records.json"

BERTSCORE_XLNET_MODEL="${PROJECT_ROOT}/BERTScore_model/xlnet-large-cased"
BERTSCORE_XLNET_NUM_LAYERS=24
BERTSCORE_XLNET_BASELINE_PATH="${BERTSCORE_XLNET_MODEL}/xlnet-large-cased.tsv"
BERTSCORE_DEVICE="cuda:0"
BERTSCORE_RESCALE_WITH_BASELINE_OPTION="--bertscore-rescale-with-baseline"

DIMENSION_LABELS=(
  "system_scope"
  "diagram_type"
  "diagram_completeness"
  "phase_regions_boundaries"
  "invariant_reactions"
)

mkdir -p "$(dirname "${OUTPUT}")"

COMMON_ARGS=(
  --metric-mode "${METRIC_MODE}"
  --prediction "${PREDICTION}"
  --ground-truth "${GROUND_TRUTH}"
  --output "${OUTPUT}"
  --bertscore-xlnet-model "${BERTSCORE_XLNET_MODEL}"
  --bertscore-xlnet-num-layers "${BERTSCORE_XLNET_NUM_LAYERS}"
  --bertscore-xlnet-baseline-path "${BERTSCORE_XLNET_BASELINE_PATH}"
  --bertscore-device "${BERTSCORE_DEVICE}"
  "${BERTSCORE_RESCALE_WITH_BASELINE_OPTION}"
  --dimension-labels "${DIMENSION_LABELS[@]}"
)

python "${PROJECT_ROOT}/src/MatPhaseBench_task_evaluation.py" \
  "${COMMON_ARGS[@]}"
