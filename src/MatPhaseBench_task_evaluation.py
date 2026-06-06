import argparse
import inspect
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from evaluation_metrics import (
    BERTSCORE_MODEL_CONFIGS,
    compute_bertscore,
    compute_llm_as_judge_metrics,
    compute_multilabel_recall_error,
    compute_numerical_prediction_error,
    compute_rouge,
)
import evaluation_metrics


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parent
    / "image_match_evaluation_data_complete_info_200_classified.json"
)
DEFAULT_LOG_OUTPUT_DIR = ROOT_DIR / "log_storage" / "llm_phase_evaluation"

MODE_DIMENSIONS = "dimensions"
METRIC_MODE_TRADITIONAL = "traditional"
METRIC_MODE_SAMPLE_DETAILS = "sample_details"
METRIC_MODE_LLM_AS_JUDGE = "llm_as_judge"
METRIC_MODE_NUMERICAL_PREDICTION_ERROR = "numerical_prediction_error"
METRIC_MODE_MULTILABEL_RECALL_ERROR = "multi_label_recall_error"
LLM_AS_JUDGE_BACKEND = "dashscope"

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
LLM_AS_JUDGE_METRIC_NAMES = ("completeness", "accuracy", "factuality")
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
    "system_scope",
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
        required=True,
        help="Prediction JSON file.",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH_PATH),
        help="Ground-truth/source JSON file.",
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
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--llm-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
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
            "Please update src/llm_phase_evaluation/evaluation_metrics.py and "
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


def load_json_list(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"JSON top level must be a list: {path}")

    return data


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_sample_id(record, fallback_index):
    if isinstance(record, dict) and "sample_id" in record:
        return record["sample_id"]
    return fallback_index


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
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not isinstance(text, str):
        return ""
    return text.strip()


def extract_ground_truth_text(record):
    text = record.get("ground_truth")
    if not isinstance(text, str):
        return ""
    return text.strip()


def extract_paper_description_text(record):
    text = record.get("paper_description")
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text if item is not None)
    elif text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    return text.strip()


def compute_sample_metrics(
    ground_truth_text,
    prediction_text,
    bertscore_lang,
    bertscore_models,
    bertscore_device,
    bertscore_rescale_with_baseline,
):
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
        metrics[bertscore_model["recall_metric_name"]] = bertscore_result["recall"]
        metrics[bertscore_model["metric_name"]] = bertscore_result["f1"]

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
):
    prediction_text = extract_prediction_text(prediction_record, mode)
    ground_truth_text = extract_ground_truth_text(ground_truth_record)

    if not prediction_text or not ground_truth_text:
        return {
            "result_type": "skipped",
            "sample_id": sample_id,
            "prediction_empty": not bool(prediction_text),
            "ground_truth_empty": not bool(ground_truth_text),
        }

    metrics = compute_sample_metrics(
        ground_truth_text=ground_truth_text,
        prediction_text=prediction_text,
        bertscore_lang=bertscore_lang,
        bertscore_models=bertscore_models,
        bertscore_device=bertscore_device,
        bertscore_rescale_with_baseline=bertscore_rescale_with_baseline,
    )
    return {
        "result_type": "sample",
        "sample_id": sample_id,
        "metrics": metrics,
        "prediction_text_length": len(prediction_text),
        "ground_truth_text_length": len(ground_truth_text),
    }


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

    selection_path = Path(path)
    with selection_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

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


def compute_traditional_metrics(
    prediction_records,
    ground_truth_records,
    mode,
    bertscore_lang="en",
    bertscore_models=BERTSCORE_MODELS,
    bertscore_device=None,
    bertscore_rescale_with_baseline=False,
):
    pairs, pairing_strategy = pair_records(prediction_records, ground_truth_records)
    sample_results = []
    skipped_samples = []

    for sample_id, prediction_record, ground_truth_record in tqdm(
        pairs,
        desc="Computing traditional text metrics",
        unit="sample",
    ):
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
            raise ValueError(f"Unsupported traditional result type: {result_type}")

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


def compute_llm_as_judge_evaluation(
    prediction_records,
    ground_truth_records,
    mode,
    llm_model=None,
    llm_api_key=None,
    llm_max_tokens=8192,
    llm_enable_thinking=False,
):
    pairs, pairing_strategy = pair_records(prediction_records, ground_truth_records)
    sample_results = []
    skipped_samples = []
    failed_samples = []

    def compute_llm_as_judge_sample_result(
        sample_id,
        prediction_record,
        ground_truth_record,
    ):
        prediction_text = extract_prediction_text(prediction_record, mode)
        paper_description = extract_paper_description_text(ground_truth_record)

        if not prediction_text or not paper_description:
            return {
                "result_type": "skipped",
                "sample_id": sample_id,
                "prediction_empty": not bool(prediction_text),
                "paper_description_empty": not bool(paper_description),
            }

        try:
            judge_metrics = compute_llm_as_judge_metrics(
                paper_description=paper_description,
                vlm_description=prediction_text,
                backend=LLM_AS_JUDGE_BACKEND,
                model=llm_model,
                api_key=llm_api_key,
                base_url=None,
                max_tokens=llm_max_tokens,
                enable_thinking=llm_enable_thinking,
            )
        except Exception as error:
            return {
                "result_type": "failed",
                "sample_id": sample_id,
                "error": str(error),
                "prediction_text_length": len(prediction_text),
                "paper_description_length": len(paper_description),
            }

        metrics = {
            metric_name: judge_metrics[metric_name]
            for metric_name in LLM_AS_JUDGE_METRIC_NAMES
        }
        return {
            "result_type": "sample",
            "sample_id": sample_id,
            "metrics": metrics,
            "prediction_text_length": len(prediction_text),
            "ground_truth_text_length": len(paper_description),
            "judge_result": judge_metrics["judge_result"],
            "finish_reason": judge_metrics.get("finish_reason"),
            "usage": judge_metrics.get("usage"),
        }

    for sample_id, prediction_record, ground_truth_record in tqdm(
        pairs,
        desc="Computing LLM-as-Judge metrics",
        unit="sample",
    ):
        result = compute_llm_as_judge_sample_result(
            sample_id,
            prediction_record,
            ground_truth_record,
        )
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
        "pairing_strategy": pairing_strategy,
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
    if "requested_sample_ids" in result:
        records[0]["requested_sample_ids"] = result["requested_sample_ids"]
    if result.get("mode") is not None:
        records[0]["mode"] = result["mode"]
    if "bertscore_models" in result:
        records[0]["bertscore_models"] = result["bertscore_models"]
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
        if "judge_result" in sample_result:
            record["judge_result"] = sample_result["judge_result"]
        if "finish_reason" in sample_result:
            record["finish_reason"] = sample_result["finish_reason"]
        if "usage" in sample_result:
            record["usage"] = sample_result["usage"]
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


def main():
    args = parse_args()
    mode = MODE_DIMENSIONS
    output_mode = mode
    prediction_path = Path(args.prediction)
    ground_truth_path = Path(args.ground_truth)
    output_path = Path(args.output) if args.output else resolve_default_output_path(args.metric_mode)

    prediction_records = load_json_list(prediction_path)
    bertscore_models = build_bertscore_models(args)
    print(f"Using evaluation_metrics from: {evaluation_metrics.__file__}", flush=True)
    dimension_aggregation_ground_truth_records = None

    if args.metric_mode == METRIC_MODE_TRADITIONAL:
        ground_truth_records = load_json_list(ground_truth_path)
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
        )
        metric_category = "traditional_automatic_metrics"
        metric_names = list(METRIC_NAMES)
        extra_metadata = {
            "bertscore_models": list(bertscore_models),
        }
    elif args.metric_mode == METRIC_MODE_SAMPLE_DETAILS:
        ground_truth_records = load_json_list(ground_truth_path)
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
        ground_truth_records = load_json_list(ground_truth_path)
        dimension_aggregation_ground_truth_records = ground_truth_records
        result = compute_llm_as_judge_evaluation(
            prediction_records=prediction_records,
            ground_truth_records=ground_truth_records,
            mode=mode,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_max_tokens=args.llm_max_tokens,
            llm_enable_thinking=args.llm_enable_thinking,
        )
        metric_category = "llm_as_judge_metrics"
        metric_names = list(LLM_AS_JUDGE_METRIC_NAMES)
        extra_metadata = {
            "llm_as_judge_config": {
                "backend": LLM_AS_JUDGE_BACKEND,
                "model": args.llm_model,
                "max_tokens": args.llm_max_tokens,
                "enable_thinking": args.llm_enable_thinking,
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
        ground_truth_path = None
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
        ground_truth_path = None
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
        "prediction_file": str(prediction_path.resolve()),
        "ground_truth_file": str(ground_truth_path.resolve()) if ground_truth_path else None,
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
    write_json(output_data, output_path)
    print(f"Wrote {args.metric_mode} metrics log to {output_path.resolve()}")


if __name__ == "__main__":
    main()
