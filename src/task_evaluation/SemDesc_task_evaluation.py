import argparse
import inspect
import math
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation_metrics import (
    BERTSCORE_MODEL_CONFIGS,
    compute_bertscore,
    compute_bertscore_batch,
    compute_llm_as_judge_metrics,
    compute_multilabel_recall_error,
    compute_numerical_prediction_error,
    compute_rouge,
)
import evaluation_metrics


DEFAULT_LOG_OUTPUT_DIR = ROOT_DIR / "log_storage" / "llm_phase_evaluation"
SEMDESC_DATASET_DIR = ROOT_DIR / "dataset" / "MatPhaseBench-SemDesc"

from utils import read_json, save_to_json
from src.llm_service import preview_text

MODE_DIMENSIONS = "dimensions"
MODE_UNLIMITED = "unlimited"
METRIC_MODE_TRADITIONAL = "traditional"
METRIC_MODE_SAMPLE_DETAILS = "sample_details"
METRIC_MODE_LLM_AS_JUDGE = "llm_as_judge"
METRIC_MODE_NUMERICAL_PREDICTION_ERROR = "numerical_prediction_error"
METRIC_MODE_MULTILABEL_RECALL_ERROR = "multi_label_recall_error"
DEFAULT_LLM_AS_JUDGE_BACKEND = "dashscope"
LLM_AS_JUDGE_BACKENDS = (
    "local",
    "dashscope",
    "jiekou",
    "icompify",
    "zhipu",
    "tokenriver",
    "tencent",
    "CNIC",
)

BASE_METRIC_NAMES = (
    "rouge1_recall",
    "rouge1_f1",
    "rougeL_recall",
    "rougeL_f1",
)
BERTSCORE_LANG = "en"
BERTSCORE_MODELS = (
    {
        "metric_name": "bertscore_xlnet_f1",
        "recall_metric_name": "bertscore_xlnet_recall",
        "model_type": str(ROOT_DIR / "BERTScore_model" / "xlnet-large-cased"),
        "num_layers": BERTSCORE_MODEL_CONFIGS["xlnet"]["num_layers"],
        "baseline_path": str(
            ROOT_DIR / "BERTScore_model" / "xlnet-large-cased" / "xlnet-large-cased.tsv"
        ),
    },
)
BERTSCORE_DEVICE = "cuda:0"
BERTSCORE_RESCALE_WITH_BASELINE = False
METRIC_NAMES = BASE_METRIC_NAMES + tuple(
    metric_name
    for bertscore_model in BERTSCORE_MODELS
    for metric_name in (
        bertscore_model["recall_metric_name"],
        bertscore_model["metric_name"],
    )
)
LLM_AS_JUDGE_METRIC_NAMES = (
    "semantic_completeness",
    "information_consistency",
    "visual_content_alignment",
    "total_score",
)
NUMERICAL_PREDICTION_ERROR_METRIC_NAMES = (
    "RNSS_DEPLOT",
    "mae",
    "rmse",
    "relaxed_accuracy_5pct",
)
MULTILABEL_RECALL_ERROR_METRIC_NAMES = (
    "macro_recall",
    "micro_recall",
)
METRIC_STAT_DIGITS = 3
DIMENSION_FIELD = "dimension_multi_classification"
SEMDESC_DIMENSION_LABELS = (
    "materials_system",
    "diagram_type",
    "diagram_completeness",
    "phase_regions_boundaries",
    "invariant_reactions",
)
DIMENSION_CONFIDENCE_LEVEL = 0.95
NORMAL_APPROX_Z_95 = 1.959963984540054


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VLM phase-diagram outputs with traditional text metrics, "
            "LLM-as-Judge metrics, or numerical prediction error metrics."
        )
    )
    parser.add_argument(
        "--metric-mode",
        choices=(
            METRIC_MODE_TRADITIONAL,
            METRIC_MODE_SAMPLE_DETAILS,
            METRIC_MODE_LLM_AS_JUDGE,
            METRIC_MODE_NUMERICAL_PREDICTION_ERROR,
            METRIC_MODE_MULTILABEL_RECALL_ERROR,
        ),
        default=METRIC_MODE_TRADITIONAL,
    )
    parser.add_argument(
        "--prediction",
        nargs="+",
        required=True,
        help="Prediction JSON file(s). Multiple files are merged before evaluation.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help=(
            "Image directory for relative image_path values in llm_as_judge mode. "
            "When omitted, paths are resolved under dataset/MatPhaseBench-SemDesc."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON records path.",
    )
    parser.add_argument(
        "--skip-dimension-aggregation",
        action="store_true",
        help="Do not write semantic-dimension score aggregation.",
    )
    parser.add_argument(
        "--dimension-labels",
        nargs="+",
        default=list(SEMDESC_DIMENSION_LABELS),
        help=(
            "Semantic dimension labels used for dimension score aggregation. "
            "Defaults to the current MatPhaseBench-SemDesc dimension labels."
        ),
    )
    parser.add_argument(
        "--dimension-sample-selection",
        default=None,
        help=(
            "Optional JSON file containing representative sample_id values for "
            "each semantic dimension. When provided, dimension score aggregation "
            "uses only these sample_id values."
        ),
    )
    parser.add_argument("--bertscore-lang", default=BERTSCORE_LANG)
    parser.add_argument("--bertscore-device", default=BERTSCORE_DEVICE)
    parser.add_argument(
        "--bertscore-xlnet-model",
        default=BERTSCORE_MODELS[0]["model_type"],
        help="Path or model name for the XLNet BERTScore model.",
    )
    parser.add_argument(
        "--bertscore-xlnet-num-layers",
        type=int,
        default=BERTSCORE_MODELS[0]["num_layers"],
        help="Number of layers for the XLNet BERTScore model.",
    )
    parser.add_argument(
        "--bertscore-xlnet-baseline-path",
        default=BERTSCORE_MODELS[0]["baseline_path"],
        help="Path to the XLNet BERTScore baseline TSV file.",
    )
    parser.add_argument(
        "--bertscore-rescale-with-baseline",
        action=argparse.BooleanOptionalAction,
        default=BERTSCORE_RESCALE_WITH_BASELINE,
    )
    parser.add_argument(
        "--trad-workers",
        type=int,
        default=1,
        help=(
            "Traditional metric worker count. It controls concurrent ROUGE "
            "sample workers and the BERTScore batch size. Defaults to 1."
        ),
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-backend",
        choices=LLM_AS_JUDGE_BACKENDS,
        default=DEFAULT_LLM_AS_JUDGE_BACKEND,
        help="Backend used by the LLM-as-Judge model.",
    )
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument(
        "--llm-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--llm-retry-count", type=int, default=3)
    parser.add_argument("--llm-retry-sleep-seconds", type=float, default=5)
    parser.add_argument("--llm-max-workers", type=int, default=1)
    parser.add_argument("--llm-judge-detail-output", default=None, help="Output JSON file for complete per-sample LLM-as-Judge prompt results.")
    parser.add_argument(
        "--llm-partial-output",
        default=None,
        help="Partial JSON path used only in llm_as_judge mode.",
    )
    parser.add_argument(
        "--llm-reprocess-failed-from",
        default=None,
        help=(
            "Formal LLM-as-Judge metrics records file. When set in llm_as_judge "
            "mode, records whose record_type is failed_sample or skipped_sample "
            "are reprocessed and merged back into the current output."
        ),
    )
    parser.add_argument(
        "--relaxed-relative-tolerance",
        type=float,
        default=0.05,
        help="Relative tolerance for numerical relaxed accuracy.",
    )
    parser.add_argument(
        "--zero-abs-tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance used when a ground-truth numeric value is zero.",
    )
    parser.add_argument(
        "--multilabel-labels",
        nargs="*",
        default=None,
        help=(
            "Optional full label space for multi-label recall. If omitted, each "
            "sample uses the union of answer and vlm_prediction labels."
        ),
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help=(
            "Sample IDs to evaluate in sample_details mode. The output contains "
            "only per-sample metric records and does not compute averages."
        ),
    )
    return parser.parse_args()


def build_bertscore_models(args):
    return (
        {
            "metric_name": "bertscore_xlnet_f1",
            "recall_metric_name": "bertscore_xlnet_recall",
            "model_type": args.bertscore_xlnet_model,
            "num_layers": args.bertscore_xlnet_num_layers,
            "baseline_path": args.bertscore_xlnet_baseline_path,
        },
    )


def validate_bertscore_baseline_support(bertscore_models, rescale_with_baseline):
    if not rescale_with_baseline:
        return

    needs_baseline_path = any(
        bool(model_config.get("baseline_path"))
        for model_config in bertscore_models
    )
    if not needs_baseline_path:
        return

    signature = inspect.signature(compute_bertscore)
    if "baseline_path" not in signature.parameters:
        raise RuntimeError(
            "The imported compute_bertscore() does not support baseline_path. "
            "Please update src/task_evaluation/evaluation_metrics.py and "
            "remove stale __pycache__ files before running baseline rescaling."
        )


def resolve_default_output_path(metric_mode):
    if metric_mode == METRIC_MODE_TRADITIONAL:
        return DEFAULT_LOG_OUTPUT_DIR / "traditional_text_metrics_records.json"
    if metric_mode == METRIC_MODE_SAMPLE_DETAILS:
        return DEFAULT_LOG_OUTPUT_DIR / "sample_details_text_metrics_records.json"
    if metric_mode == METRIC_MODE_LLM_AS_JUDGE:
        return DEFAULT_LOG_OUTPUT_DIR / "llm_as_judge_text_metrics_records.json"
    if metric_mode == METRIC_MODE_NUMERICAL_PREDICTION_ERROR:
        return DEFAULT_LOG_OUTPUT_DIR / "numerical_prediction_error_metrics_records.json"
    if metric_mode == METRIC_MODE_MULTILABEL_RECALL_ERROR:
        return DEFAULT_LOG_OUTPUT_DIR / "multi_label_recall_error_metrics_records.json"
    raise ValueError(f"Unsupported metric_mode: {metric_mode}")


def get_sample_id(record, fallback_index):
    if isinstance(record, dict) and "sample_id" in record:
        return record["sample_id"]
    return fallback_index


def get_sample_source_fields(record):
    fields = {}
    for field_name in (
        "source_prediction_file",
        "source_prediction_file_index",
        "source_record_index",
        "source_sample_id",
    ):
        if isinstance(record, dict) and field_name in record:
            fields[field_name] = record[field_name]
    return fields


def load_prediction_records(prediction_paths):
    if not prediction_paths:
        raise ValueError("--prediction requires at least one JSON file.")

    paths = [Path(path) for path in prediction_paths]
    merged_records = []
    multi_file = len(paths) > 1

    for file_index, path in enumerate(paths, start=1):
        records = read_json(path)
        if not isinstance(records, list):
            raise ValueError(f"Prediction JSON file must contain a list: {path}")

        resolved_path = str(path.resolve())
        for record_index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                merged_records.append(record)
                continue

            updated_record = dict(record)
            source_sample_id = get_sample_id(record, record_index)
            updated_record["source_prediction_file"] = resolved_path
            updated_record["source_prediction_file_index"] = file_index
            updated_record["source_record_index"] = record_index
            updated_record["source_sample_id"] = source_sample_id
            if multi_file:
                updated_record["sample_id"] = (
                    f"{file_index}:{record_index}:{normalize_sample_id(source_sample_id)}"
                )
            merged_records.append(updated_record)

    return merged_records, [str(path.resolve()) for path in paths]


def infer_description_mode(prediction_records):
    detected_modes = set()
    for record in prediction_records:
        if not isinstance(record, dict):
            continue

        vlm_description = record.get("vlm_description")
        if not isinstance(vlm_description, dict):
            continue

        if "comprehensive_description" in vlm_description:
            detected_modes.add(MODE_DIMENSIONS)
        if "diagram_description" in vlm_description:
            detected_modes.add(MODE_UNLIMITED)

    if not detected_modes:
        raise ValueError(
            "Unable to infer description mode: no prediction record contains "
            "vlm_description.comprehensive_description or "
            "vlm_description.diagram_description."
        )
    if len(detected_modes) > 1:
        raise ValueError(
            "Prediction records mix incompatible description modes: "
            f"{sorted(detected_modes)}"
        )

    return detected_modes.pop()


def build_sample_id_index(records, name):
    sample_id_to_record = {}
    duplicate_sample_ids = []

    for index, record in enumerate(records):
        if not isinstance(record, dict) or "sample_id" not in record:
            return None, []

        sample_id = record["sample_id"]
        if sample_id in sample_id_to_record:
            duplicate_sample_ids.append(sample_id)
        sample_id_to_record[sample_id] = record

    if duplicate_sample_ids:
        raise ValueError(
            f"Duplicate sample_id values in {name}: {sorted(duplicate_sample_ids)}"
        )

    return sample_id_to_record, []


def pair_records(prediction_records, ground_truth_records):
    prediction_records = [
        record for record in prediction_records if isinstance(record, dict)
    ]
    ground_truth_records = [
        record for record in ground_truth_records if isinstance(record, dict)
    ]

    if len(prediction_records) == len(ground_truth_records):
        pairs = []
        for index, (prediction, ground_truth) in enumerate(
            zip(prediction_records, ground_truth_records)
        ):
            prediction_sample_id = get_sample_id(prediction, index)
            ground_truth_sample_id = get_sample_id(ground_truth, index)
            if prediction_sample_id != ground_truth_sample_id:
                raise ValueError(
                    "sample_id mismatch at index "
                    f"{index}: prediction={prediction_sample_id}, "
                    f"ground_truth={ground_truth_sample_id}"
                )
            pairs.append((prediction_sample_id, prediction, ground_truth))
        return pairs, "sequence"

    prediction_index, _ = build_sample_id_index(prediction_records, "prediction")
    ground_truth_index, _ = build_sample_id_index(ground_truth_records, "ground_truth")
    if prediction_index is None or ground_truth_index is None:
        raise ValueError(
            "Prediction and ground-truth record counts differ, and sample_id "
            "pairing is unavailable."
        )

    missing_ground_truth_ids = [
        sample_id
        for sample_id in prediction_index
        if sample_id not in ground_truth_index
    ]
    if missing_ground_truth_ids:
        raise ValueError(
            "Prediction contains sample_id values missing from ground truth: "
            f"{missing_ground_truth_ids}"
        )

    pairs = [
        (prediction["sample_id"], prediction, ground_truth_index[prediction["sample_id"]])
        for prediction in prediction_records
    ]
    return pairs, "sample_id"


def extract_prediction_text(record, mode):
    if mode == MODE_DIMENSIONS:
        vlm_description = record.get("vlm_description")
        text = (
            vlm_description.get("comprehensive_description")
            if isinstance(vlm_description, dict)
            else None
        )
    elif mode == MODE_UNLIMITED:
        vlm_description = record.get("vlm_description")
        text = (
            vlm_description.get("diagram_description")
            if isinstance(vlm_description, dict)
            else None
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not isinstance(text, str):
        return ""
    return text.strip()


def extract_ground_truth_text(record):
    text = record.get("llm_description")
    if not isinstance(text, str):
        return ""
    return text.strip()


def build_ground_truth_records_from_predictions(prediction_records):
    if not isinstance(prediction_records, list):
        raise ValueError("Prediction JSON file must contain a list of records.")

    ground_truth_records = []
    for index, record in enumerate(prediction_records, start=1):
        if not isinstance(record, dict):
            continue

        ground_truth = record.get("ground_truth")
        if isinstance(ground_truth, list):
            ground_truth = "\n".join(
                str(item) for item in ground_truth if item is not None
            )
        elif ground_truth is None:
            ground_truth = ""
        elif not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)
        ground_truth = ground_truth.strip()

        selected_dimensions = record.get("selected_dimensions")
        if not isinstance(selected_dimensions, list):
            selected_dimensions = []

        ground_truth_records.append(
            {
                "sample_id": get_sample_id(record, index),
                "llm_description": ground_truth,
                DIMENSION_FIELD: {"labels": selected_dimensions},
                **get_sample_source_fields(record),
            }
        )

    return ground_truth_records


def compute_sample_rouge_metrics(ground_truth_text, prediction_text):
    rouge_result = compute_rouge(
        ground_truth=ground_truth_text,
        generated_text=prediction_text,
        rouge_types=["rouge1", "rougeL"],
        score_types=["recall", "f1"],
    )
    metrics = {
        "rouge1_recall": rouge_result["rouge1"]["recall"],
        "rouge1_f1": rouge_result["rouge1"]["f1"],
        "rougeL_recall": rouge_result["rougeL"]["recall"],
        "rougeL_f1": rouge_result["rougeL"]["f1"],
    }
    return metrics


def compute_traditional_sample_result(
    sample_id,
    prediction_record,
    ground_truth_record,
    mode,
    bertscore_lang,
    bertscore_models,
    bertscore_device,
    bertscore_rescale_with_baseline,
    include_bertscore=True,
    keep_text_for_batch_bertscore=False,
):
    prediction_text = extract_prediction_text(prediction_record, mode)
    ground_truth_text = extract_ground_truth_text(ground_truth_record)

    if not prediction_text or not ground_truth_text:
        return {
            "result_type": "skipped",
            "sample_id": sample_id,
            "prediction_empty": not bool(prediction_text),
            "ground_truth_empty": not bool(ground_truth_text),
            **get_sample_source_fields(prediction_record),
        }

    metrics = compute_sample_rouge_metrics(ground_truth_text, prediction_text)
    if include_bertscore:
        for bertscore_model in bertscore_models:
            bertscore_result = compute_bertscore(
                ground_truth=ground_truth_text,
                generated_text=prediction_text,
                score_types=["recall", "f1"],
                lang=bertscore_lang,
                model_type=bertscore_model["model_type"],
                num_layers=bertscore_model.get("num_layers"),
                device=bertscore_device,
                rescale_with_baseline=bertscore_rescale_with_baseline,
                baseline_path=bertscore_model.get("baseline_path"),
            )
            metrics[bertscore_model["recall_metric_name"]] = bertscore_result[
                "recall"
            ]
            metrics[bertscore_model["metric_name"]] = bertscore_result["f1"]

    result = {
        "result_type": "sample",
        "sample_id": sample_id,
        "metrics": metrics,
        "prediction_text_length": len(prediction_text),
        "ground_truth_text_length": len(ground_truth_text),
        **get_sample_source_fields(prediction_record),
    }
    if keep_text_for_batch_bertscore:
        result["_prediction_text"] = prediction_text
        result["_ground_truth_text"] = ground_truth_text
    return result


def average_metrics(sample_results, metric_names=METRIC_NAMES):
    if not sample_results:
        return {metric_name: 0.0 for metric_name in metric_names}

    summary = {}
    for metric_name in metric_names:
        values = [
            result["metrics"][metric_name]
            for result in sample_results
            if metric_name in result.get("metrics", {})
            and result["metrics"][metric_name] is not None
        ]
        summary[metric_name] = sum(values) / len(values) if values else 0.0

    return summary


def compute_metric_statistics(sample_results, metric_names):
    statistics = {}
    for metric_name in metric_names:
        values = [
            float(result["metrics"][metric_name])
            for result in sample_results
            if metric_name in result.get("metrics", {})
            and result["metrics"][metric_name] is not None
        ]
        if not values:
            mean_value = 0.0
            std_value = 0.0
        else:
            mean_value = sum(values) / len(values)
            std_value = math.sqrt(
                sum((value - mean_value) ** 2 for value in values) / len(values)
            )

        mean_value = round(mean_value, METRIC_STAT_DIGITS)
        std_value = round(std_value, METRIC_STAT_DIGITS)
        statistics[metric_name] = {
            "mean": mean_value,
            "std": std_value,
            "mean_std": (
                f"{mean_value:.{METRIC_STAT_DIGITS}f} +/- "
                f"{std_value:.{METRIC_STAT_DIGITS}f}"
            ),
            "sample_count": len(values),
        }

    return statistics


def normalize_sample_id(sample_id):
    if isinstance(sample_id, bool):
        return str(sample_id)
    if isinstance(sample_id, int):
        return str(sample_id)
    if isinstance(sample_id, float) and sample_id.is_integer():
        return str(int(sample_id))
    return str(sample_id).strip()


def sort_sample_ids(sample_ids):
    def sample_id_sort_key(value):
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    return sorted(sample_ids, key=sample_id_sort_key)


def get_dimension_labels(record, dimension_field=DIMENSION_FIELD):
    classification = record.get(dimension_field)
    if not isinstance(classification, dict):
        return None

    labels = classification.get("labels")
    if not isinstance(labels, list):
        return None

    normalized_labels = []
    seen_labels = set()
    for label in labels:
        if not isinstance(label, str):
            continue

        label = label.strip()
        if not label or label in seen_labels:
            continue

        normalized_labels.append(label)
        seen_labels.add(label)

    return normalized_labels


def build_dimension_sample_index(
    ground_truth_records,
    dimension_labels=SEMDESC_DIMENSION_LABELS,
    dimension_field=DIMENSION_FIELD,
    selected_dimension_sample_ids=None,
):
    dimension_label_set = set(dimension_labels)
    dimension_to_sample_ids = {label: [] for label in dimension_labels}
    missing_or_invalid_dimension_samples = []
    samples_with_non_current_dimension_labels = []
    non_current_dimension_label_occurrence_count = 0

    for index, record in enumerate(ground_truth_records, start=1):
        if not isinstance(record, dict):
            continue

        sample_id = get_sample_id(record, index)
        normalized_sample_id = normalize_sample_id(sample_id)
        labels = get_dimension_labels(record, dimension_field)
        if labels is None:
            missing_or_invalid_dimension_samples.append(sample_id)
            continue

        has_non_current_label = False
        for label in labels:
            if label not in dimension_label_set:
                non_current_dimension_label_occurrence_count += 1
                has_non_current_label = True
                continue

            dimension_to_sample_ids[label].append(normalized_sample_id)

        if has_non_current_label:
            samples_with_non_current_dimension_labels.append(sample_id)

    for dimension in dimension_to_sample_ids:
        dimension_to_sample_ids[dimension] = sort_sample_ids(
            set(dimension_to_sample_ids[dimension])
        )

    selection_missing_sample_ids = {}
    if selected_dimension_sample_ids is not None:
        for dimension in dimension_to_sample_ids:
            available_sample_ids = set(dimension_to_sample_ids[dimension])
            selected_sample_ids = [
                normalize_sample_id(sample_id)
                for sample_id in selected_dimension_sample_ids.get(dimension, [])
            ]
            selected_sample_ids = sort_sample_ids(set(selected_sample_ids))
            missing_sample_ids = [
                sample_id
                for sample_id in selected_sample_ids
                if sample_id not in available_sample_ids
            ]
            if missing_sample_ids:
                selection_missing_sample_ids[dimension] = missing_sample_ids

            dimension_to_sample_ids[dimension] = [
                sample_id
                for sample_id in selected_sample_ids
                if sample_id in available_sample_ids
            ]

    return {
        "dimension_to_sample_ids": dimension_to_sample_ids,
        "missing_or_invalid_dimension_samples": sort_sample_ids(
            missing_or_invalid_dimension_samples
        ),
        "samples_with_non_current_dimension_labels": sort_sample_ids(
            set(samples_with_non_current_dimension_labels)
        ),
        "non_current_dimension_label_occurrence_count": (
            non_current_dimension_label_occurrence_count
        ),
        "selection_missing_sample_ids": selection_missing_sample_ids,
    }


def build_sample_result_metric_index(sample_results):
    sample_metric_index = {}
    duplicate_sample_ids = []
    for sample_result in sample_results:
        sample_id = sample_result.get("sample_id")
        normalized_sample_id = normalize_sample_id(sample_id)
        if normalized_sample_id in sample_metric_index:
            duplicate_sample_ids.append(sample_id)

        sample_metric_index[normalized_sample_id] = sample_result.get("metrics", {})

    if duplicate_sample_ids:
        raise ValueError(
            f"Duplicate sample_id values in sample_results: {duplicate_sample_ids}"
        )

    return sample_metric_index


def load_dimension_sample_selection(path, dimension_labels):
    if path is None:
        return None

    data = read_json(path)
    selection_path = Path(path)

    if not isinstance(data, dict):
        raise ValueError(
            f"Dimension sample selection file must contain a JSON object: {path}"
        )

    raw_selection = data.get("selection")
    if raw_selection is None:
        raw_selection = data.get("dimension_to_sample_ids")
    if not isinstance(raw_selection, dict):
        raise ValueError(
            "Dimension sample selection file must contain a 'selection' object "
            "or a 'dimension_to_sample_ids' object."
        )

    selected_dimension_sample_ids = {}
    for dimension in dimension_labels:
        dimension_selection = raw_selection.get(dimension)
        if isinstance(dimension_selection, dict):
            sample_ids = dimension_selection.get("sample_ids")
        else:
            sample_ids = dimension_selection

        if sample_ids is None:
            selected_dimension_sample_ids[dimension] = []
            continue
        if not isinstance(sample_ids, list):
            raise ValueError(
                f"Dimension sample selection for {dimension} must be a list "
                "or an object with a sample_ids list."
            )

        selected_dimension_sample_ids[dimension] = sort_sample_ids(
            set(normalize_sample_id(sample_id) for sample_id in sample_ids)
        )

    return {
        "path": str(selection_path.resolve()),
        "record_type": data.get("record_type"),
        "dimension_field": data.get("dimension_field"),
        "samples_per_dimension": data.get("samples_per_dimension"),
        "random_seed": data.get("random_seed"),
        "selected_dimension_sample_ids": selected_dimension_sample_ids,
    }


def compute_dimension_metric_statistics(values):
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    sample_count = len(numeric_values)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "confidence_level": DIMENSION_CONFIDENCE_LEVEL,
            "confidence_interval": None,
        }

    mean_value = statistics.fmean(numeric_values)
    std_value = (
        statistics.pstdev(numeric_values)
        if sample_count == 1
        else statistics.stdev(numeric_values)
    )
    standard_error = std_value / math.sqrt(sample_count)
    margin = NORMAL_APPROX_Z_95 * standard_error

    return {
        "sample_count": sample_count,
        "mean": mean_value,
        "std": std_value,
        "median": statistics.median(numeric_values),
        "min": min(numeric_values),
        "max": max(numeric_values),
        "confidence_level": DIMENSION_CONFIDENCE_LEVEL,
        "confidence_interval": {
            "low": mean_value - margin,
            "high": mean_value + margin,
            "method": "normal_approximation",
        },
    }


def build_dimension_score_aggregation(
    result,
    ground_truth_records,
    dimension_labels=SEMDESC_DIMENSION_LABELS,
    dimension_field=DIMENSION_FIELD,
    dimension_sample_selection=None,
):
    selected_dimension_sample_ids = (
        dimension_sample_selection.get("selected_dimension_sample_ids")
        if isinstance(dimension_sample_selection, dict)
        else None
    )
    dimension_sample_index = build_dimension_sample_index(
        ground_truth_records=ground_truth_records,
        dimension_labels=dimension_labels,
        dimension_field=dimension_field,
        selected_dimension_sample_ids=selected_dimension_sample_ids,
    )
    sample_metric_index = build_sample_result_metric_index(result["sample_results"])
    metric_names = result["metric_names"]

    dimension_results = {}
    missing_metric_sample_ids = {}
    for dimension, sample_ids in dimension_sample_index[
        "dimension_to_sample_ids"
    ].items():
        metric_values = {metric_name: [] for metric_name in metric_names}
        matched_sample_ids = []
        missing_for_dimension = []

        for sample_id in sample_ids:
            sample_metrics = sample_metric_index.get(sample_id)
            if sample_metrics is None:
                missing_for_dimension.append(sample_id)
                continue

            matched_sample_ids.append(sample_id)
            for metric_name in metric_names:
                value = sample_metrics.get(metric_name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_values[metric_name].append(float(value))

        if missing_for_dimension:
            missing_metric_sample_ids[dimension] = missing_for_dimension

        dimension_results[dimension] = {
            "classified_sample_count": len(sample_ids),
            "evaluated_sample_count": len(matched_sample_ids),
            "sample_ids": matched_sample_ids,
            "metric_values": metric_values,
            "metric_statistics": {
                metric_name: compute_dimension_metric_statistics(values)
                for metric_name, values in metric_values.items()
            },
        }

    return {
        "generated_at": result["generated_at"],
        "prediction_file": result["prediction_file"],
        "ground_truth_file": result["ground_truth_file"],
        "metric_category": result["metric_category"],
        "dimension_field": dimension_field,
        "dimension_labels": list(dimension_labels),
        "dimension_sample_selection": {
            key: value
            for key, value in dimension_sample_selection.items()
            if key != "selected_dimension_sample_ids"
        }
        if isinstance(dimension_sample_selection, dict)
        else None,
        "metric_names": metric_names,
        "dimension_results": dimension_results,
        "missing_metric_sample_ids": missing_metric_sample_ids,
        **{
            key: value
            for key, value in dimension_sample_index.items()
            if key != "dimension_to_sample_ids"
        },
    }


def attach_batch_bertscore_metrics(
    sample_results,
    bertscore_lang,
    bertscore_models,
    bertscore_device,
    bertscore_rescale_with_baseline,
    batch_size,
):
    if not sample_results:
        return

    ground_truth_texts = [
        sample_result["_ground_truth_text"] for sample_result in sample_results
    ]
    prediction_texts = [
        sample_result["_prediction_text"] for sample_result in sample_results
    ]

    for bertscore_model in bertscore_models:
        bertscore_results = compute_bertscore_batch(
            ground_truths=ground_truth_texts,
            generated_texts=prediction_texts,
            score_types=["recall", "f1"],
            lang=bertscore_lang,
            model_type=bertscore_model["model_type"],
            num_layers=bertscore_model.get("num_layers"),
            device=bertscore_device,
            batch_size=batch_size,
            rescale_with_baseline=bertscore_rescale_with_baseline,
            baseline_path=bertscore_model.get("baseline_path"),
        )
        for sample_result, bertscore_result in zip(
            sample_results,
            bertscore_results,
        ):
            sample_result["metrics"][bertscore_model["recall_metric_name"]] = (
                bertscore_result["recall"]
            )
            sample_result["metrics"][bertscore_model["metric_name"]] = (
                bertscore_result["f1"]
            )

    for sample_result in sample_results:
        sample_result.pop("_ground_truth_text", None)
        sample_result.pop("_prediction_text", None)


def compute_traditional_metrics(
    prediction_records,
    ground_truth_records,
    mode,
    bertscore_lang="en",
    bertscore_models=BERTSCORE_MODELS,
    bertscore_device=None,
    bertscore_rescale_with_baseline=False,
    trad_workers=1,
):
    if trad_workers < 1:
        raise ValueError("trad_workers must be >= 1.")

    pairs, pairing_strategy = pair_records(prediction_records, ground_truth_records)
    sample_results = []
    skipped_samples = []

    def compute_pair(pair):
        sample_id, prediction_record, ground_truth_record = pair
        result = compute_traditional_sample_result(
            sample_id=sample_id,
            prediction_record=prediction_record,
            ground_truth_record=ground_truth_record,
            mode=mode,
            bertscore_lang=bertscore_lang,
            bertscore_models=bertscore_models,
            bertscore_device=bertscore_device,
            bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
            include_bertscore=False,
            keep_text_for_batch_bertscore=True,
        )
        return result

    def collect_result(result):
        result_type = result.pop("result_type")
        if result_type == "sample":
            sample_results.append(result)
        elif result_type == "skipped":
            skipped_samples.append(result)
        else:
            raise ValueError(f"Unsupported traditional result type: {result_type}")

    if trad_workers == 1:
        for pair in tqdm(
            pairs,
            desc="Computing traditional text metrics",
            unit="sample",
        ):
            collect_result(compute_pair(pair))
    else:
        completed_results = {}
        with ThreadPoolExecutor(max_workers=trad_workers) as executor:
            future_to_index = {
                executor.submit(compute_pair, pair): index
                for index, pair in enumerate(pairs)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Computing traditional text metrics",
                unit="sample",
            ):
                completed_results[future_to_index[future]] = future.result()

        for index in sorted(completed_results):
            collect_result(completed_results[index])

    attach_batch_bertscore_metrics(
        sample_results=sample_results,
        bertscore_lang=bertscore_lang,
        bertscore_models=bertscore_models,
        bertscore_device=bertscore_device,
        bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
        batch_size=trad_workers,
    )

    return {
        "pairing_strategy": pairing_strategy,
        "evaluated_sample_count": len(sample_results),
        "skipped_sample_count": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "summary": average_metrics(sample_results, METRIC_NAMES),
        "metric_statistics": compute_metric_statistics(sample_results, METRIC_NAMES),
        "sample_results": sample_results,
    }


def compute_selected_sample_details(
    prediction_records,
    ground_truth_records,
    mode,
    sample_ids,
    bertscore_lang="en",
    bertscore_models=BERTSCORE_MODELS,
    bertscore_device=None,
    bertscore_rescale_with_baseline=False,
):
    if not sample_ids:
        raise ValueError("--sample-ids is required in sample_details mode.")

    pairs, pairing_strategy = pair_records(prediction_records, ground_truth_records)
    pair_index = {}
    duplicate_sample_ids = []
    for sample_id, prediction_record, ground_truth_record in pairs:
        normalized_sample_id = normalize_sample_id(sample_id)
        if normalized_sample_id in pair_index:
            duplicate_sample_ids.append(sample_id)
        pair_index[normalized_sample_id] = (
            sample_id,
            prediction_record,
            ground_truth_record,
        )

    if duplicate_sample_ids:
        raise ValueError(
            f"Duplicate paired sample_id values: {duplicate_sample_ids}"
        )

    requested_sample_ids = [normalize_sample_id(sample_id) for sample_id in sample_ids]
    missing_sample_ids = [
        sample_id for sample_id in requested_sample_ids if sample_id not in pair_index
    ]
    if missing_sample_ids:
        raise ValueError(
            "Requested sample IDs are missing from paired prediction/ground truth: "
            f"{missing_sample_ids}"
        )

    sample_results = []
    skipped_samples = []
    for requested_sample_id in tqdm(
        requested_sample_ids,
        desc="Computing selected sample text metrics",
        unit="sample",
    ):
        sample_id, prediction_record, ground_truth_record = pair_index[
            requested_sample_id
        ]
        result = compute_traditional_sample_result(
            sample_id=sample_id,
            prediction_record=prediction_record,
            ground_truth_record=ground_truth_record,
            mode=mode,
            bertscore_lang=bertscore_lang,
            bertscore_models=bertscore_models,
            bertscore_device=bertscore_device,
            bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
        )
        result_type = result.pop("result_type")
        if result_type == "sample":
            sample_results.append(result)
        elif result_type == "skipped":
            skipped_samples.append(result)
        else:
            raise ValueError(f"Unsupported sample-details result type: {result_type}")

    return {
        "pairing_strategy": pairing_strategy,
        "requested_sample_ids": requested_sample_ids,
        "evaluated_sample_count": len(sample_results),
        "skipped_sample_count": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "sample_results": sample_results,
    }


def save_llm_as_judge_partial_results(completed_results, partial_output_path):
    if partial_output_path is None:
        return

    partial_output_path = Path(partial_output_path)
    partial_output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        completed_results[index]
        for index in sorted(completed_results)
    ]
    save_to_json(records, partial_output_path, output_info=False)
    print(
        f"\nSaved partial LLM-as-Judge evaluation result "
        f"({len(records)} records) to {partial_output_path.resolve()}",
        flush=True,
    )


def load_llm_as_judge_partial_results(prediction_records, partial_output_path):
    if partial_output_path is None:
        return {}

    partial_output_path = Path(partial_output_path)
    if not partial_output_path.exists():
        return {}

    records = read_json(partial_output_path)
    if not isinstance(records, list):
        raise ValueError(f"Partial output must contain a list: {partial_output_path}")

    completed_results = {}
    for record in records:
        if not isinstance(record, dict):
            continue

        input_index = record.get("input_index")
        if not isinstance(input_index, int):
            raise ValueError(
                f"Partial record missing integer input_index: {partial_output_path}"
            )
        if input_index < 0 or input_index >= len(prediction_records):
            raise ValueError(
                f"Partial record input_index out of range: {input_index}"
            )

        current_sample = prediction_records[input_index]
        current_sample_id = (
            get_sample_id(current_sample, input_index + 1)
            if isinstance(current_sample, dict)
            else input_index + 1
        )
        if record.get("sample_id") != current_sample_id:
            raise ValueError(
                "Partial output sample_id mismatch: "
                f"input_index={input_index}, partial={record.get('sample_id')}, "
                f"current={current_sample_id}"
            )
        if (
            record.get("result_type") == "sample"
            and not isinstance(record.get("judge_output"), dict)
        ):
            continue

        completed_results[input_index] = record

    print(f"Resuming from partial output: {partial_output_path.resolve()}", flush=True)
    return completed_results


def remove_llm_as_judge_partial_output(partial_output_path):
    if partial_output_path is None:
        return

    partial_output_path = Path(partial_output_path)
    if partial_output_path.exists():
        partial_output_path.unlink()
        print(f"Removed partial output: {partial_output_path.resolve()}", flush=True)


def build_llm_as_judge_error_detail(error):
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None

    return {
        "backend": response.get("backend"),
        "model": response.get("model"),
        "error": response.get("error"),
        "raw_text": response.get("raw_text"),
        "raw_text_preview": preview_text(response.get("raw_text")),
        "reasoning": response.get("reasoning"),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "raw_response": response.get("raw_response"),
        "is_multimodal": response.get("is_multimodal"),
        "image": response.get("image"),
    }


def resolve_llm_as_judge_image_path(image_path, image_dir=None):
    image_path = image_path.strip()
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", image_path):
        return image_path

    path = Path(image_path).expanduser()
    if path.is_absolute():
        return str(path)

    if image_dir:
        return str((Path(image_dir).expanduser() / image_path).resolve())

    candidates = [
        SEMDESC_DATASET_DIR / image_path,
        SEMDESC_DATASET_DIR / "phase_images" / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def get_source_model_error(vlm_description):
    if vlm_description == "Abnormal":
        return {"type": "abnormal_marker"}
    if isinstance(vlm_description, dict) and str(vlm_description.get("code")) == "403":
        return {
            "type": "source_model_http_error",
            "code": 403,
            "reason": vlm_description.get("reason"),
            "message": vlm_description.get("message"),
        }
    return None


def load_failed_sample_ids_from_llm_as_judge_log(path):
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"LLM-as-Judge metrics log must contain a list: {path}")

    failed_sample_ids = []
    seen_sample_ids = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("record_type") not in {"failed_sample", "skipped_sample"}:
            continue

        sample_id = record.get("sample_id")
        normalized_sample_id = normalize_sample_id(sample_id)
        if not normalized_sample_id or normalized_sample_id in seen_sample_ids:
            continue

        failed_sample_ids.append(normalized_sample_id)
        seen_sample_ids.add(normalized_sample_id)

    return failed_sample_ids


def filter_prediction_records_by_sample_ids(prediction_records, sample_ids):
    selected_sample_ids = set(sample_ids)
    filtered_records = []
    matched_sample_ids = set()

    for index, record in enumerate(prediction_records, start=1):
        if not isinstance(record, dict):
            continue

        sample_id = get_sample_id(record, index)
        normalized_sample_id = normalize_sample_id(sample_id)
        if normalized_sample_id in selected_sample_ids:
            filtered_records.append(record)
            matched_sample_ids.add(normalized_sample_id)

    missing_sample_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in matched_sample_ids
    ]
    return filtered_records, missing_sample_ids


def collect_llm_as_judge_result_from_records(records, metric_names):
    sample_results = []
    skipped_samples = []
    failed_samples = []
    for record in records:
        if not isinstance(record, dict):
            continue

        record_type = record.get("record_type")
        if record_type == "sample_metric":
            sample_result = {
                "sample_id": record.get("sample_id"),
                "metrics": {
                    metric_name: record[metric_name]
                    for metric_name in metric_names
                    if metric_name in record
                },
            }
            for field_name in (
                "input_index",
                "image_path",
                "prediction_text_length",
                "ground_truth_text_length",
                "source_prediction_file",
                "source_prediction_file_index",
                "source_record_index",
                "source_sample_id",
            ):
                if field_name in record:
                    sample_result[field_name] = record[field_name]
            sample_results.append(sample_result)
        elif record_type == "skipped_sample":
            skipped_samples.append(
                {key: value for key, value in record.items() if key != "record_type"}
            )
        elif record_type == "failed_sample":
            failed_samples.append(
                {key: value for key, value in record.items() if key != "record_type"}
            )

    return sample_results, skipped_samples, failed_samples


def merge_llm_as_judge_reprocess_result(
    previous_records,
    reprocess_result,
    reprocessed_sample_ids,
    metric_names,
):
    reprocessed_sample_ids = set(reprocessed_sample_ids)
    previous_sample_results, previous_skipped_samples, previous_failed_samples = (
        collect_llm_as_judge_result_from_records(previous_records, metric_names)
    )

    def not_reprocessed(record):
        return normalize_sample_id(record.get("sample_id")) not in reprocessed_sample_ids

    sample_results = [
        record for record in previous_sample_results if not_reprocessed(record)
    ] + reprocess_result["sample_results"]
    skipped_samples = [
        record for record in previous_skipped_samples if not_reprocessed(record)
    ] + reprocess_result["skipped_samples"]
    failed_samples = [
        record for record in previous_failed_samples if not_reprocessed(record)
    ] + reprocess_result.get("failed_samples", [])

    return {
        "pairing_strategy": "llm_as_judge_failed_sample_reprocess_from_formal_records",
        "evaluated_sample_count": len(sample_results),
        "skipped_sample_count": len(skipped_samples),
        "failed_sample_count": len(failed_samples),
        "skipped_samples": skipped_samples,
        "failed_samples": failed_samples,
        "summary": average_metrics(sample_results, metric_names),
        "metric_statistics": compute_metric_statistics(sample_results, metric_names),
        "sample_results": sample_results,
    }


def compute_llm_as_judge_evaluation(
    prediction_records,
    llm_backend=DEFAULT_LLM_AS_JUDGE_BACKEND,
    llm_model=None,
    llm_api_key=None,
    llm_enable_thinking=False,
    retry_count=3,
    retry_sleep_seconds=5,
    llm_max_workers=1,
    partial_output_path=None,
    image_dir=None,
):
    if not isinstance(prediction_records, list):
        raise ValueError("prediction_records must be a list.")
    if retry_count < 1:
        raise ValueError("retry_count must be >= 1.")
    if retry_sleep_seconds < 0:
        raise ValueError("retry_sleep_seconds must be >= 0.")
    if llm_max_workers < 1:
        raise ValueError("llm_max_workers must be >= 1.")

    completed_results = load_llm_as_judge_partial_results(
        prediction_records,
        partial_output_path,
    )

    def compute_llm_as_judge_sample_result(input_index, sample):
        source_fields = get_sample_source_fields(sample)
        if not isinstance(sample, dict):
            return {
                "result_type": "skipped",
                "input_index": input_index,
                "sample_id": input_index + 1,
                "invalid_sample": True,
            }

        sample_id = get_sample_id(sample, input_index + 1)
        ground_truth = sample.get("ground_truth")
        vlm_description = sample.get("vlm_description")
        source_model_error = get_source_model_error(vlm_description)
        prediction_text = (
            vlm_description.get("comprehensive_description")
            if isinstance(vlm_description, dict)
            else None
        )
        selected_dimensions = sample.get("selected_dimensions")
        image_path = sample.get("image_path")

        missing_fields = []
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            missing_fields.append("ground_truth")
        if not isinstance(prediction_text, str) or not prediction_text.strip():
            missing_fields.append("vlm_description.comprehensive_description")
        if not isinstance(selected_dimensions, list) or not selected_dimensions:
            missing_fields.append("selected_dimensions")
        if not isinstance(image_path, str) or not image_path.strip():
            missing_fields.append("image_path")
        if missing_fields:
            skipped_result = {
                "result_type": "skipped",
                "input_index": input_index,
                "sample_id": sample_id,
                "missing_fields": missing_fields,
                **source_fields,
            }
            if source_model_error is not None:
                skipped_result["source_model_error"] = source_model_error
            return skipped_result

        image_path = resolve_llm_as_judge_image_path(image_path, image_dir)

        sample = {
            **sample,
            "sample_id": sample_id,
            "image_path": image_path,
        }
        for attempt in range(1, retry_count + 1):
            try:
                judge_metrics = compute_llm_as_judge_metrics(
                    sample=sample,
                    backend=llm_backend,
                    model=llm_model,
                    api_key=llm_api_key,
                    base_url=None,
                    enable_thinking=llm_enable_thinking,
                    include_judge_output=True,
                )
                break
            except Exception as error:
                if attempt >= retry_count:
                    return {
                        "result_type": "failed",
                        "input_index": input_index,
                        "sample_id": sample_id,
                        "error": str(error),
                        "llm_error_detail": build_llm_as_judge_error_detail(error),
                        "attempt_count": attempt,
                        "prediction_text_length": len(prediction_text),
                        "ground_truth_text_length": len(ground_truth),
                        **source_fields,
                    }

                if retry_sleep_seconds:
                    time.sleep(retry_sleep_seconds)

        metrics = {
            metric_name: judge_metrics[metric_name]
            for metric_name in LLM_AS_JUDGE_METRIC_NAMES
        }
        return {
            "result_type": "sample",
            "input_index": input_index,
            "sample_id": sample_id,
            "metrics": metrics,
            "judge_output": judge_metrics["judge_output"],
            "prediction_text_length": len(prediction_text),
            "ground_truth_text_length": len(ground_truth),
            "image_path": image_path,
            **source_fields,
        }

    pending_items = [
        (input_index, sample)
        for input_index, sample in enumerate(prediction_records)
        if input_index not in completed_results
    ]
    if pending_items:
        with ThreadPoolExecutor(max_workers=llm_max_workers) as executor:
            future_to_input_index = {
                executor.submit(
                    compute_llm_as_judge_sample_result,
                    input_index,
                    sample,
                ): input_index
                for input_index, sample in pending_items
            }
            for future in tqdm(
                as_completed(future_to_input_index),
                total=len(future_to_input_index),
                desc="Computing LLM-as-Judge metrics",
                unit="sample",
            ):
                input_index = future_to_input_index[future]
                completed_results[input_index] = future.result()
                save_llm_as_judge_partial_results(
                    completed_results,
                    partial_output_path,
                )
    else:
        print("All LLM-as-Judge samples were loaded from partial output.", flush=True)

    sample_results = []
    skipped_samples = []
    failed_samples = []
    for input_index in sorted(completed_results):
        result = dict(completed_results[input_index])
        result_type = result.pop("result_type")
        if result_type == "sample":
            sample_results.append(result)
        elif result_type == "skipped":
            skipped_samples.append(result)
        elif result_type == "failed":
            failed_samples.append(result)
        else:
            raise ValueError(f"Unsupported LLM-as-Judge result type: {result_type}")

    return {
        "pairing_strategy": "prediction_records_with_embedded_ground_truth",
        "evaluated_sample_count": len(sample_results),
        "skipped_sample_count": len(skipped_samples),
        "failed_sample_count": len(failed_samples),
        "skipped_samples": skipped_samples,
        "failed_samples": failed_samples,
        "summary": average_metrics(sample_results, LLM_AS_JUDGE_METRIC_NAMES),
        "metric_statistics": compute_metric_statistics(
            sample_results,
            LLM_AS_JUDGE_METRIC_NAMES,
        ),
        "sample_results": sample_results,
    }


def extract_numeric_sequence(record, field_name):
    value = record.get(field_name)
    if isinstance(value, dict):
        if "coordinate_values" in value:
            value = value["coordinate_values"]
        elif "values" in value:
            value = value["values"]

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")

    return value


def extract_label_sequence(record, field_name):
    value = record.get(field_name)
    if isinstance(value, dict):
        if "reactions" in value:
            value = value["reactions"]
        elif "invariant_reaction_entities" in value:
            value = value["invariant_reaction_entities"]
        elif "labels" in value:
            value = value["labels"]

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")

    return value


def count_multilabel_recall_components(ground_truth_labels, predicted_labels, labels=None):
    if labels is None:
        selected_labels = set(ground_truth_labels) | set(predicted_labels)
    else:
        selected_labels = set(labels)

    ground_truth_set = {
        str(label).strip()
        for label in ground_truth_labels
        if str(label).strip() in selected_labels
    }
    prediction_set = {
        str(label).strip()
        for label in predicted_labels
        if str(label).strip() in selected_labels
    }
    true_positive_count = len(ground_truth_set & prediction_set)
    false_negative_count = len(ground_truth_set - prediction_set)
    return true_positive_count, false_negative_count


def compute_numerical_prediction_error_evaluation(
    prediction_records,
    relaxed_relative_tolerance=0.05,
    zero_abs_tolerance=1e-12,
):
    prediction_records = [
        record for record in prediction_records if isinstance(record, dict)
    ]
    sample_results = []
    skipped_samples = []

    for index, record in enumerate(
        tqdm(
            prediction_records,
            desc="Computing numerical prediction error metrics",
            unit="sample",
        ),
        start=1,
    ):
        sample_id = get_sample_id(record, index)
        try:
            ground_truth_values = extract_numeric_sequence(record, "answer")
            predicted_values = extract_numeric_sequence(record, "vlm_prediction")
            metrics = compute_numerical_prediction_error(
                ground_truth_values=ground_truth_values,
                predicted_values=predicted_values,
                relaxed_relative_tolerance=relaxed_relative_tolerance,
                zero_abs_tolerance=zero_abs_tolerance,
            )
        except Exception as error:
            skipped_samples.append(
                {
                    "sample_id": sample_id,
                    "error": str(error),
                }
            )
            continue

        sample_results.append(
            {
                "sample_id": sample_id,
                "image_path": record.get("image_path"),
                "metrics": metrics,
                "ground_truth_value_count": len(ground_truth_values),
                "prediction_value_count": len(predicted_values),
            }
        )

    return {
        "pairing_strategy": "single_file",
        "evaluated_sample_count": len(sample_results),
        "skipped_sample_count": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "summary": average_metrics(
            sample_results,
            NUMERICAL_PREDICTION_ERROR_METRIC_NAMES,
        ),
        "metric_statistics": compute_metric_statistics(
            sample_results,
            NUMERICAL_PREDICTION_ERROR_METRIC_NAMES,
        ),
        "sample_results": sample_results,
    }


def compute_multilabel_recall_error_evaluation(
    prediction_records,
    labels=None,
):
    prediction_records = [
        record for record in prediction_records if isinstance(record, dict)
    ]
    sample_results = []
    skipped_samples = []
    total_true_positive_count = 0
    total_false_negative_count = 0

    for index, record in enumerate(
        tqdm(
            prediction_records,
            desc="Computing multi-label recall error metrics",
            unit="sample",
        ),
        start=1,
    ):
        sample_id = get_sample_id(record, index)
        try:
            ground_truth_labels = extract_label_sequence(record, "answer")
            predicted_labels = extract_label_sequence(record, "vlm_prediction")
            metrics = compute_multilabel_recall_error(
                ground_truth_labels=ground_truth_labels,
                predicted_labels=predicted_labels,
                labels=labels,
            )
            true_positive_count, false_negative_count = (
                count_multilabel_recall_components(
                    ground_truth_labels,
                    predicted_labels,
                    labels=labels,
                )
            )
        except Exception as error:
            skipped_samples.append(
                {
                    "sample_id": sample_id,
                    "error": str(error),
                }
            )
            continue

        total_true_positive_count += true_positive_count
        total_false_negative_count += false_negative_count
        sample_results.append(
            {
                "sample_id": sample_id,
                "image_path": record.get("image_path"),
                "metrics": metrics,
                "ground_truth_label_count": len(ground_truth_labels),
                "prediction_label_count": len(predicted_labels),
                "true_positive_count": true_positive_count,
                "false_negative_count": false_negative_count,
            }
        )

    summary = average_metrics(
        sample_results,
        MULTILABEL_RECALL_ERROR_METRIC_NAMES,
    )
    micro_denominator = total_true_positive_count + total_false_negative_count
    summary["micro_recall"] = (
        total_true_positive_count / micro_denominator
        if micro_denominator
        else 1.0
    )

    return {
        "pairing_strategy": "single_file",
        "evaluated_sample_count": len(sample_results),
        "skipped_sample_count": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "summary": summary,
        "metric_statistics": compute_metric_statistics(
            sample_results,
            MULTILABEL_RECALL_ERROR_METRIC_NAMES,
        ),
        "total_true_positive_count": total_true_positive_count,
        "total_false_negative_count": total_false_negative_count,
        "sample_results": sample_results,
    }


def build_json_records(result):
    records = [
        {
            "record_type": "metadata",
            "generated_at": result["generated_at"],
            "metric_category": result["metric_category"],
            "prediction_file": result["prediction_file"],
            "ground_truth_file": result["ground_truth_file"],
            "pairing_strategy": result["pairing_strategy"],
            "evaluated_sample_count": result["evaluated_sample_count"],
            "skipped_sample_count": result["skipped_sample_count"],
            "metric_names": result["metric_names"],
        },
    ]
    if "summary" in result:
        records[0]["summary"] = result["summary"]
    if "metric_statistics" in result:
        records[0]["metric_statistics"] = result["metric_statistics"]
    if "prediction_files" in result:
        records[0]["prediction_files"] = result["prediction_files"]
        records[0]["prediction_file_count"] = len(result["prediction_files"])
    if "requested_sample_ids" in result:
        records[0]["requested_sample_ids"] = result["requested_sample_ids"]
    if result.get("mode") is not None:
        records[0]["mode"] = result["mode"]
    if "bertscore_models" in result:
        records[0]["bertscore_models"] = result["bertscore_models"]
    if "traditional_config" in result:
        records[0]["traditional_config"] = result["traditional_config"]
    if "llm_as_judge_config" in result:
        records[0]["llm_as_judge_config"] = result["llm_as_judge_config"]
    if "numerical_prediction_error_config" in result:
        records[0]["numerical_prediction_error_config"] = (
            result["numerical_prediction_error_config"]
        )
    if "multi_label_recall_error_config" in result:
        records[0]["multi_label_recall_error_config"] = (
            result["multi_label_recall_error_config"]
        )
    if "failed_sample_count" in result:
        records[0]["failed_sample_count"] = result["failed_sample_count"]
    if "total_true_positive_count" in result:
        records[0]["total_true_positive_count"] = result["total_true_positive_count"]
    if "total_false_negative_count" in result:
        records[0]["total_false_negative_count"] = result["total_false_negative_count"]

    for sample_result in result["sample_results"]:
        record = {
            "record_type": "sample_metric",
            "sample_id": sample_result["sample_id"],
            **sample_result["metrics"],
        }
        if "image_path" in sample_result:
            record["image_path"] = sample_result["image_path"]
        for field_name in (
            "source_prediction_file",
            "source_prediction_file_index",
            "source_record_index",
            "source_sample_id",
        ):
            if field_name in sample_result:
                record[field_name] = sample_result[field_name]
        if "prediction_text_length" in sample_result:
            record["prediction_text_length"] = sample_result["prediction_text_length"]
        if "ground_truth_text_length" in sample_result:
            record["ground_truth_text_length"] = sample_result["ground_truth_text_length"]
        if "ground_truth_value_count" in sample_result:
            record["ground_truth_value_count"] = sample_result["ground_truth_value_count"]
        if "prediction_value_count" in sample_result:
            record["prediction_value_count"] = sample_result["prediction_value_count"]
        if "ground_truth_label_count" in sample_result:
            record["ground_truth_label_count"] = sample_result["ground_truth_label_count"]
        if "prediction_label_count" in sample_result:
            record["prediction_label_count"] = sample_result["prediction_label_count"]
        if "true_positive_count" in sample_result:
            record["true_positive_count"] = sample_result["true_positive_count"]
        if "false_negative_count" in sample_result:
            record["false_negative_count"] = sample_result["false_negative_count"]
        records.append(record)

    for skipped_sample in result["skipped_samples"]:
        records.append(
            {
                "record_type": "skipped_sample",
                **skipped_sample,
            }
        )

    for failed_sample in result.get("failed_samples", []):
        records.append(
            {
                "record_type": "failed_sample",
                **failed_sample,
            }
        )

    if "dimension_score_aggregation" in result:
        records.append(
            {
                "record_type": "dimension_score_aggregation",
                **result["dimension_score_aggregation"],
            }
        )

    return records


def build_llm_as_judge_detail_records(result, previous_records=None):
    previous_records_by_sample_id = {
        normalize_sample_id(record.get("sample_id")): record
        for record in previous_records or []
        if isinstance(record, dict)
        and record.get("record_type") == "sample_judge_result"
        and record.get("sample_id") is not None
    }
    records = []
    for sample_result in result["sample_results"]:
        judge_output = sample_result.get("judge_output")
        if not isinstance(judge_output, dict):
            previous_record = previous_records_by_sample_id.get(
                normalize_sample_id(sample_result.get("sample_id"))
            )
            if previous_record is None:
                raise ValueError(
                    "Missing complete LLM-as-Judge output for sample: "
                    f"{sample_result.get('sample_id')}"
                )
            records.append(previous_record)
            continue

        record = {
            "record_type": "sample_judge_result",
            "sample_id": sample_result["sample_id"],
        }
        if "image_path" in sample_result:
            record["image_path"] = sample_result["image_path"]
        for field_name in (
            "source_prediction_file",
            "source_prediction_file_index",
            "source_record_index",
            "source_sample_id",
            "prediction_text_length",
            "ground_truth_text_length",
        ):
            if field_name in sample_result:
                record[field_name] = sample_result[field_name]
        record.update(judge_output)
        records.append(record)

    for skipped_sample in result["skipped_samples"]:
        records.append({"record_type": "skipped_sample", **skipped_sample})
    for failed_sample in result.get("failed_samples", []):
        records.append({"record_type": "failed_sample", **failed_sample})
    return records


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else resolve_default_output_path(args.metric_mode)

    prediction_records, prediction_files = load_prediction_records(args.prediction)
    if args.metric_mode in (
        METRIC_MODE_TRADITIONAL,
        METRIC_MODE_SAMPLE_DETAILS,
        METRIC_MODE_LLM_AS_JUDGE,
    ):
        mode = infer_description_mode(prediction_records)
    else:
        mode = None
    output_mode = mode
    prediction_file = prediction_files[0] if len(prediction_files) == 1 else None
    bertscore_models = build_bertscore_models(args)
    print(f"Using evaluation_metrics from: {evaluation_metrics.__file__}", flush=True)
    print(
        f"Loaded {len(prediction_records)} prediction records from "
        f"{len(prediction_files)} file(s).",
        flush=True,
    )
    dimension_aggregation_ground_truth_records = None
    llm_judge_detail_output_path = None
    previous_llm_as_judge_detail_records = None

    if args.metric_mode == METRIC_MODE_TRADITIONAL:
        ground_truth_records = build_ground_truth_records_from_predictions(
            prediction_records
        )
        if mode == MODE_DIMENSIONS:
            dimension_aggregation_ground_truth_records = ground_truth_records
        print(
            f"compute_bertscore signature: {inspect.signature(compute_bertscore)}",
            flush=True,
        )
        validate_bertscore_baseline_support(
            bertscore_models,
            args.bertscore_rescale_with_baseline,
        )
        result = compute_traditional_metrics(
            prediction_records=prediction_records,
            ground_truth_records=ground_truth_records,
            mode=mode,
            bertscore_lang=args.bertscore_lang,
            bertscore_models=bertscore_models,
            bertscore_device=args.bertscore_device,
            bertscore_rescale_with_baseline=args.bertscore_rescale_with_baseline,
            trad_workers=args.trad_workers,
        )
        metric_category = "traditional_automatic_metrics"
        metric_names = list(METRIC_NAMES)
        extra_metadata = {
            "bertscore_models": list(bertscore_models),
            "traditional_config": {
                "rouge_workers": args.trad_workers,
                "bertscore_batch_size": args.trad_workers,
            },
        }
    elif args.metric_mode == METRIC_MODE_SAMPLE_DETAILS:
        ground_truth_records = build_ground_truth_records_from_predictions(
            prediction_records
        )
        print(
            f"compute_bertscore signature: {inspect.signature(compute_bertscore)}",
            flush=True,
        )
        validate_bertscore_baseline_support(
            bertscore_models,
            args.bertscore_rescale_with_baseline,
        )
        result = compute_selected_sample_details(
            prediction_records=prediction_records,
            ground_truth_records=ground_truth_records,
            mode=mode,
            sample_ids=args.sample_ids,
            bertscore_lang=args.bertscore_lang,
            bertscore_models=bertscore_models,
            bertscore_device=args.bertscore_device,
            bertscore_rescale_with_baseline=args.bertscore_rescale_with_baseline,
        )
        metric_category = "selected_sample_traditional_metrics"
        metric_names = list(METRIC_NAMES)
        extra_metadata = {
            "bertscore_models": list(bertscore_models),
        }
    elif args.metric_mode == METRIC_MODE_LLM_AS_JUDGE:
        if mode == MODE_UNLIMITED:
            raise ValueError(
                "llm_as_judge does not support unlimited descriptions yet. "
                "Use --metric-mode traditional or sample_details."
            )
        llm_partial_output_path = (
            Path(args.llm_partial_output)
            if args.llm_partial_output
            else output_path.with_name(f"{output_path.stem}.partial.json")
        )
        llm_judge_detail_output_path = (
            Path(args.llm_judge_detail_output)
            if args.llm_judge_detail_output
            else output_path.with_name(f"{output_path.stem}.prompt_records.json")
        )
        reprocess_failed_sample_ids = []
        reprocess_missing_sample_ids = []
        previous_llm_as_judge_records = None
        if args.llm_reprocess_failed_from:
            previous_llm_as_judge_records = read_json(args.llm_reprocess_failed_from)
            if not llm_judge_detail_output_path.exists():
                raise FileNotFoundError(
                    "Detailed LLM-as-Judge output is required when reprocessing "
                    f"failed samples: {llm_judge_detail_output_path}"
                )
            previous_llm_as_judge_detail_records = read_json(
                llm_judge_detail_output_path
            )
            if not isinstance(previous_llm_as_judge_detail_records, list):
                raise ValueError(
                    "Detailed LLM-as-Judge output must contain a list: "
                    f"{llm_judge_detail_output_path}"
                )
            reprocess_failed_sample_ids = load_failed_sample_ids_from_llm_as_judge_log(
                args.llm_reprocess_failed_from
            )
            prediction_records, reprocess_missing_sample_ids = (
                filter_prediction_records_by_sample_ids(
                    prediction_records,
                    reprocess_failed_sample_ids,
                )
            )
            print(
                "Reprocessing failed or skipped LLM-as-Judge samples: "
                f"requested={len(reprocess_failed_sample_ids)}, "
                f"matched={len(prediction_records)}, "
                f"missing={len(reprocess_missing_sample_ids)}",
                flush=True,
            )

        ground_truth_records = build_ground_truth_records_from_predictions(
            prediction_records
        )
        dimension_aggregation_ground_truth_records = ground_truth_records
        result = compute_llm_as_judge_evaluation(
            prediction_records=prediction_records,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_enable_thinking=args.llm_enable_thinking,
            retry_count=args.llm_retry_count,
            retry_sleep_seconds=args.llm_retry_sleep_seconds,
            llm_max_workers=args.llm_max_workers,
            partial_output_path=llm_partial_output_path,
            image_dir=args.image_dir,
        )
        if previous_llm_as_judge_records is not None:
            result = merge_llm_as_judge_reprocess_result(
                previous_records=previous_llm_as_judge_records,
                reprocess_result=result,
                reprocessed_sample_ids=reprocess_failed_sample_ids,
                metric_names=LLM_AS_JUDGE_METRIC_NAMES,
            )

        metric_category = "llm_as_judge_metrics"
        metric_names = list(LLM_AS_JUDGE_METRIC_NAMES)
        extra_metadata = {
            "llm_as_judge_config": {
                "backend": args.llm_backend,
                "model": args.llm_model,
                "enable_thinking": args.llm_enable_thinking,
                "retry_count": args.llm_retry_count,
                "retry_sleep_seconds": args.llm_retry_sleep_seconds,
                "max_workers": args.llm_max_workers,
                "image_dir": (
                    str(Path(args.image_dir).expanduser().resolve())
                    if args.image_dir
                    else None
                ),
                "partial_output": str(llm_partial_output_path.resolve()),
                "judge_detail_output": str(
                    llm_judge_detail_output_path.resolve()
                ),
                "reprocess_failed_from": (
                    str(Path(args.llm_reprocess_failed_from).resolve())
                    if args.llm_reprocess_failed_from
                    else None
                ),
                "reprocess_failed_sample_ids": reprocess_failed_sample_ids,
                "reprocess_missing_sample_ids": reprocess_missing_sample_ids,
            }
        }
    elif args.metric_mode == METRIC_MODE_NUMERICAL_PREDICTION_ERROR:
        result = compute_numerical_prediction_error_evaluation(
            prediction_records=prediction_records,
            relaxed_relative_tolerance=args.relaxed_relative_tolerance,
            zero_abs_tolerance=args.zero_abs_tolerance,
        )
        metric_category = "numerical_prediction_error_metrics"
        metric_names = list(NUMERICAL_PREDICTION_ERROR_METRIC_NAMES)
        output_mode = None
        extra_metadata = {
            "numerical_prediction_error_config": {
                "relaxed_relative_tolerance": args.relaxed_relative_tolerance,
                "zero_abs_tolerance": args.zero_abs_tolerance,
            }
        }
    elif args.metric_mode == METRIC_MODE_MULTILABEL_RECALL_ERROR:
        result = compute_multilabel_recall_error_evaluation(
            prediction_records=prediction_records,
            labels=args.multilabel_labels,
        )
        metric_category = "multi_label_recall_error_metrics"
        metric_names = list(MULTILABEL_RECALL_ERROR_METRIC_NAMES)
        output_mode = None
        extra_metadata = {
            "multi_label_recall_error_config": {
                "labels": args.multilabel_labels,
            }
        }
    else:
        raise ValueError(f"Unsupported metric mode: {args.metric_mode}")

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metric_category": metric_category,
        "mode": output_mode,
        "prediction_file": prediction_file,
        "prediction_files": prediction_files,
        "ground_truth_file": None,
        "metric_names": metric_names,
        **extra_metadata,
        **result,
    }

    if (
        dimension_aggregation_ground_truth_records is not None
        and not args.skip_dimension_aggregation
    ):
        dimension_sample_selection = load_dimension_sample_selection(
            args.dimension_sample_selection,
            args.dimension_labels,
        )
        dimension_score_aggregation = build_dimension_score_aggregation(
            result=result,
            ground_truth_records=dimension_aggregation_ground_truth_records,
            dimension_labels=args.dimension_labels,
            dimension_sample_selection=dimension_sample_selection,
        )
        result["dimension_score_aggregation"] = dimension_score_aggregation

    output_data = build_json_records(result)
    save_to_json(output_data, output_path, output_info=False)
    if args.metric_mode == METRIC_MODE_LLM_AS_JUDGE:
        detail_output_data = build_llm_as_judge_detail_records(
            result,
            previous_records=previous_llm_as_judge_detail_records,
        )
        save_to_json(
            detail_output_data,
            llm_judge_detail_output_path,
            output_info=False,
        )
        remove_llm_as_judge_partial_output(llm_partial_output_path)
        print(
            "Wrote complete LLM-as-Judge prompt records to "
            f"{llm_judge_detail_output_path.resolve()}"
        )
    print(f"Wrote {args.metric_mode} metrics log to {output_path.resolve()}")


if __name__ == "__main__":
    main()
