from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

DEFAULT_ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
DEFAULT_SCORE_TYPES = ("precision", "recall", "f1")
LLM_AS_JUDGE_DIMENSIONS = ("completeness", "accuracy", "factuality")
BERTSCORE_MODEL_CONFIGS = {
    "scibert": {
        "model_type": "allenai/scibert_scivocab_cased",
        "num_layers": 12,
    },
    "deberta_xlarge_mnli": {
        "model_type": "microsoft/deberta-xlarge-mnli",
        "num_layers": 24,
    },
    "xlnet": {
        "model_type": "xlnet-large-cased",
        "num_layers": 24,
    },
}


MODULE_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[1]
for import_dir in (SRC_DIR, MODULE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from llm_service import llm_infer, require_llm_json_answer
# from llm_as_judge_prompt import judge_prompt


def _ensure_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _normalize_names(
    values: str | Iterable[str] | None,
    allowed_values: Iterable[str],
    default_values: Iterable[str],
    name: str,
) -> list[str]:
    if values is None:
        values = default_values
    elif isinstance(values, str):
        values = [values]

    normalized = []
    seen = set()
    allowed_values = tuple(allowed_values)
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} entries must be strings.")

        if value not in allowed_values:
            raise ValueError(
                f"Unsupported {name}: {value}. "
                f"Allowed values: {list(allowed_values)}"
            )

        if value not in seen:
            seen.add(value)
            normalized.append(value)

    if not normalized:
        raise ValueError(f"{name} cannot be empty.")

    return normalized


def _select_scores(score_values: dict[str, float], score_types: Iterable[str]) -> dict[str, float]:
    return {score_type: score_values[score_type] for score_type in score_types}


def _to_float_list(values: Sequence[Any], name: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numeric values.")

    numeric_values = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise TypeError(f"{name}[{index}] must be numeric, got bool.")
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name}[{index}] must be numeric: {value!r}") from error

    return numeric_values


def _to_label_list(values: Sequence[Any], name: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of labels.")

    labels = []
    seen = set()
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a string label: {value!r}")

        label = value.strip()
        if not label:
            continue
        if label not in seen:
            seen.add(label)
            labels.append(label)

    return labels


def _relative_distance(target: float, prediction: float, zero_abs_tolerance: float) -> float:
    if target == 0:
        return 0.0 if abs(prediction) <= zero_abs_tolerance else 1.0
    return min(abs(target - prediction) / abs(target), 1.0)


def _match_numeric_values(
    ground_truth_values: list[float],
    predicted_values: list[float],
    zero_abs_tolerance: float,
) -> list[tuple[int, int, float]]:
    if not ground_truth_values or not predicted_values:
        return []

    target_count = len(ground_truth_values)
    prediction_count = len(predicted_values)
    state_count = 1 << prediction_count
    dp: dict[tuple[int, int], tuple[float, tuple[tuple[int, int, float], ...]]] = {
        (0, 0): (0.0, tuple())
    }

    for target_index, target in enumerate(ground_truth_values):
        next_dp = dict(dp)
        for (used_mask, matched_count), (cost, pairs) in dp.items():
            for prediction_index, prediction in enumerate(predicted_values):
                bit = 1 << prediction_index
                if used_mask & bit:
                    continue

                distance = _relative_distance(target, prediction, zero_abs_tolerance)
                next_key = (used_mask | bit, matched_count + 1)
                next_value = (
                    cost + distance,
                    pairs + ((target_index, prediction_index, distance),),
                )
                current_value = next_dp.get(next_key)
                if current_value is None or next_value[0] < current_value[0]:
                    next_dp[next_key] = next_value
        dp = next_dp

    expected_match_count = min(target_count, prediction_count)
    best_pairs: tuple[tuple[int, int, float], ...] = tuple()
    best_cost = math.inf
    for (_used_mask, matched_count), (cost, pairs) in dp.items():
        if matched_count != expected_match_count:
            continue
        if cost < best_cost:
            best_cost = cost
            best_pairs = pairs

    return list(best_pairs)


def compute_numerical_prediction_error(
    ground_truth_values: Sequence[Any],
    predicted_values: Sequence[Any],
    *,
    relaxed_relative_tolerance: float = 0.05,
    zero_abs_tolerance: float = 1e-12,
) -> dict[str, float | None]:
    """Compute one-sample numerical list prediction errors.

    The function matches predicted values to ground-truth values with a global
    minimum relative-error assignment, so the input lists do not need to have the
    same length or order.

    Returns:
        {
          "RNSS_DEPLOT": 0.9,
          "mae": 10.0,
          "rmse": 12.3,
          "relaxed_accuracy_5pct": 0.667
        }
    """

    ground_truth_values = _to_float_list(ground_truth_values, "ground_truth_values")
    predicted_values = _to_float_list(predicted_values, "predicted_values")

    target_count = len(ground_truth_values)
    prediction_count = len(predicted_values)
    max_count = max(target_count, prediction_count)
    if max_count == 0:
        return {
            "RNSS_DEPLOT": 1.0,
            "mae": 0.0,
            "rmse": 0.0,
            "relaxed_accuracy_5pct": 1.0,
        }

    matched_pairs = _match_numeric_values(
        ground_truth_values,
        predicted_values,
        zero_abs_tolerance,
    )
    unmatched_count = max_count - len(matched_pairs)

    relative_distance_sum = sum(distance for _target_idx, _pred_idx, distance in matched_pairs)
    rnss_deplot = 1.0 - (relative_distance_sum + unmatched_count) / max_count
    rnss_deplot = max(0.0, min(1.0, rnss_deplot))

    if not matched_pairs:
        return {
            "RNSS_DEPLOT": rnss_deplot,
            "mae": None,
            "rmse": None,
            "relaxed_accuracy_5pct": 0.0,
        }

    absolute_errors = [
        abs(ground_truth_values[target_idx] - predicted_values[pred_idx])
        for target_idx, pred_idx, _distance in matched_pairs
    ]
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(sum(error * error for error in absolute_errors) / len(absolute_errors))

    relaxed_correct_count = 0
    for target_idx, pred_idx, _distance in matched_pairs:
        target = ground_truth_values[target_idx]
        prediction = predicted_values[pred_idx]
        if target == 0:
            is_correct = abs(prediction) <= zero_abs_tolerance
        else:
            is_correct = abs(target - prediction) / abs(target) <= relaxed_relative_tolerance
        if is_correct:
            relaxed_correct_count += 1

    relaxed_accuracy = relaxed_correct_count / max_count

    return {
        "RNSS_DEPLOT": float(rnss_deplot),
        "mae": float(mae),
        "rmse": float(rmse),
        "relaxed_accuracy_5pct": float(relaxed_accuracy),
    }


def compute_multilabel_recall_error(
    ground_truth_labels: Sequence[Any],
    predicted_labels: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    zero_division: int = 0,
) -> dict[str, float]:
    """Compute one-sample multi-label recall metrics.

    The function is designed for LocalReason ``reactions_prediction`` outputs,
    where one image may contain multiple invariant reaction labels.

    Returns:
        {
          "macro_recall": 0.5,
          "micro_recall": 1.0
        }
    """

    ground_truth_labels = _to_label_list(ground_truth_labels, "ground_truth_labels")
    predicted_labels = _to_label_list(predicted_labels, "predicted_labels")

    if labels is None:
        selected_labels = sorted(set(ground_truth_labels) | set(predicted_labels))
    else:
        selected_labels = _to_label_list(labels, "labels")

    if not selected_labels:
        return {
            "macro_recall": 1.0,
            "micro_recall": 1.0,
        }

    try:
        from sklearn.metrics import recall_score
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as error:
        raise ImportError(
            "Multi-label recall computation requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        ) from error

    binarizer = MultiLabelBinarizer(classes=selected_labels)
    y_true = binarizer.fit_transform([ground_truth_labels])
    y_pred = binarizer.transform([predicted_labels])

    return {
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=zero_division,
            )
        ),
        "micro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=zero_division,
            )
        ),
    }


def compute_rouge(
    ground_truth: str,
    generated_text: str,
    rouge_types: str | Iterable[str] | None = None,
    score_types: str | Iterable[str] | None = None,
    use_stemmer: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute ROUGE scores between a reference text and generated text.

    Args:
        ground_truth: Reference text.
        generated_text: Model generated text.
        rouge_types: One or more ROUGE variants. Supported values are
            "rouge1", "rouge2", and "rougeL". Defaults to all three.
        score_types: One or more score fields: precision, recall, and/or f1.
            Defaults to all three.
        use_stemmer: Passed to rouge_score's RougeScorer.

    Returns:
        A nested dictionary, for example:
        {
          "rouge1": {"precision": 0.5, "recall": 0.4, "f1": 0.4444},
          "rougeL": {"f1": 0.3333}
        }
    """

    ground_truth = _ensure_text(ground_truth, "ground_truth")
    generated_text = _ensure_text(generated_text, "generated_text")
    selected_rouge_types = _normalize_names(
        rouge_types,
        DEFAULT_ROUGE_TYPES,
        DEFAULT_ROUGE_TYPES,
        "rouge_types",
    )
    selected_score_types = _normalize_names(
        score_types,
        DEFAULT_SCORE_TYPES,
        DEFAULT_SCORE_TYPES,
        "score_types",
    )

    try:
        from rouge_score import rouge_scorer
    except ImportError as error:
        raise ImportError(
            "ROUGE computation now uses the 'rouge-score' package directly. "
            "Install it with: pip install rouge-score"
        ) from error

    scorer = rouge_scorer.RougeScorer(
        selected_rouge_types,
        use_stemmer=use_stemmer,
    )
    detailed_scores = scorer.score(ground_truth, generated_text)

    results = {}
    for rouge_type in selected_rouge_types:
        score = detailed_scores[rouge_type]
        score_values = {
            "precision": float(score.precision),
            "recall": float(score.recall),
            "f1": float(score.fmeasure),
        }
        results[rouge_type] = _select_scores(score_values, selected_score_types)

    return results


def compute_bertscore(
    ground_truth: str,
    generated_text: str,
    score_types: str | Iterable[str] | None = None,
    lang: str = "en",
    model_type: str | None = None,
    num_layers: int | None = None,
    device: str | None = None,
    batch_size: int = 64,
    idf: bool = False,
    rescale_with_baseline: bool = False,
    baseline_path: str | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute BERTScore between a reference text and generated text.

    Args:
        ground_truth: Reference text.
        generated_text: Model generated text.
        score_types: One or more score fields: precision, recall, and/or f1.
            Defaults to all three.
        lang: Language code passed to bert_score when model_type is not specified.
        model_type: Optional BERTScore model name.
        num_layers: Optional number of model layers. Required by bert_score when
            model_type is a local model path that is not in its built-in mapping.
        device: Optional torch device string, such as "cuda:0" or "cpu".
        batch_size: Batch size passed to bert_score.
        idf: Whether to use inverse document frequency weighting.
        rescale_with_baseline: Whether to rescale with BERTScore baseline.
        baseline_path: Optional path to a custom BERTScore baseline TSV file.
        verbose: Whether BERTScore should print progress information.

    Returns:
        A flat dictionary, for example:
        {"precision": 0.91, "recall": 0.89, "f1": 0.90}
    """

    ground_truth = _ensure_text(ground_truth, "ground_truth")
    generated_text = _ensure_text(generated_text, "generated_text")
    selected_score_types = _normalize_names(
        score_types,
        DEFAULT_SCORE_TYPES,
        DEFAULT_SCORE_TYPES,
        "score_types",
    )

    try:
        from bert_score import score as bert_score_score
    except ImportError as error:
        raise ImportError(
            "BERTScore computation now uses the 'bert-score' package directly. "
            "Install it with: pip install bert-score"
        ) from error

    score_kwargs = {
        "cands": [generated_text],
        "refs": [ground_truth],
        "lang": lang,
        "batch_size": batch_size,
        "idf": idf,
        "rescale_with_baseline": rescale_with_baseline,
        "verbose": verbose,
    }
    if baseline_path:
        score_kwargs["baseline_path"] = baseline_path
    if model_type:
        score_kwargs["model_type"] = model_type
    if num_layers is not None:
        score_kwargs["num_layers"] = num_layers
    if device:
        score_kwargs["device"] = device

    precision, recall, f1 = bert_score_score(**score_kwargs)
    score_values = {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }
    return _select_scores(score_values, selected_score_types)


def compute_text_generation_metrics(
    ground_truth: str,
    generated_text: str,
    include_rouge: bool = True,
    include_bertscore: bool = False,
    rouge_types: str | Iterable[str] | None = None,
    score_types: str | Iterable[str] | None = None,
    rouge_use_stemmer: bool = True,
    bertscore_lang: str = "en",
    bertscore_model_type: str | None = None,
    bertscore_num_layers: int | None = None,
    bertscore_device: str | None = None,
    bertscore_batch_size: int = 64,
    bertscore_idf: bool = False,
    bertscore_rescale_with_baseline: bool = False,
    bertscore_baseline_path: str | None = None,
    bertscore_verbose: bool = False,
) -> dict[str, object]:
    """Compute selected text generation metrics in one call."""

    results = {}
    if include_rouge:
        results["rouge"] = compute_rouge(
            ground_truth=ground_truth,
            generated_text=generated_text,
            rouge_types=rouge_types,
            score_types=score_types,
            use_stemmer=rouge_use_stemmer,
        )

    if include_bertscore:
        results["bertscore"] = compute_bertscore(
            ground_truth=ground_truth,
            generated_text=generated_text,
            score_types=score_types,
            lang=bertscore_lang,
            model_type=bertscore_model_type,
            num_layers=bertscore_num_layers,
            device=bertscore_device,
            batch_size=bertscore_batch_size,
            idf=bertscore_idf,
            rescale_with_baseline=bertscore_rescale_with_baseline,
            baseline_path=bertscore_baseline_path,
            verbose=bertscore_verbose,
        )

    return results


def build_llm_as_judge_prompt(
    paper_description: str,
    vlm_description: str,
) -> str:
    """
    Build the prompt for one LLM-as-Judge evaluation sample.
    """
    paper_description = _ensure_text(paper_description, "paper_description")
    vlm_description = _ensure_text(vlm_description, "vlm_description")

    return judge_prompt.format(
        paper_description=paper_description,
        vlm_description=vlm_description,
    )


def _extract_dimension_score(judge_answer: dict[str, Any], dimension: str) -> int:
    dimension_result = judge_answer.get(dimension)
    if not isinstance(dimension_result, dict):
        raise ValueError(f"LLM-as-Judge output missing dimension object: {dimension}")

    score = dimension_result.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(f"LLM-as-Judge score must be numeric: {dimension}.score")

    score = int(score)
    if score < 0 or score > 5:
        raise ValueError(f"LLM-as-Judge score out of range [0, 5]: {dimension}={score}")
    return score


def extract_llm_as_judge_scores(judge_answer: dict[str, Any]) -> dict[str, int]:
    """
    Extract completeness, accuracy, and factuality scores from a judge JSON.
    """
    if not isinstance(judge_answer, dict):
        raise TypeError("judge_answer must be a dict.")

    return {
        dimension: _extract_dimension_score(judge_answer, dimension)
        for dimension in LLM_AS_JUDGE_DIMENSIONS
    }


def compute_llm_as_judge_metrics(
    paper_description: str,
    vlm_description: str,
    *,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    top_p: float = 0.8,
    max_tokens: int = 8192,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
) -> dict[str, Any]:
    """
    Compute one-sample LLM-as-Judge scores.

    Args:
        paper_description: Ground-truth/source paper text.
        vlm_description: Model-generated text to evaluate.
        backend/model/api_key/base_url: Passed to ``llm_service.llm_infer``.

    Returns:
        A dictionary containing the three scores and the full judge JSON:
        {
          "completeness": 4,
          "accuracy": 5,
          "factuality": 4,
          "judge_result": {...}
        }
    """
    prompt = build_llm_as_judge_prompt(
        paper_description=paper_description,
        vlm_description=vlm_description,
    )

    response = llm_infer(
        prompt=prompt,
        backend=backend,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        timeout=timeout,
    )
    judge_answer = require_llm_json_answer(response, stage_name="LLM-as-Judge")
    scores = extract_llm_as_judge_scores(judge_answer)

    return {
        **scores,
        "judge_result": judge_answer,
        "model": response.get("model"),
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
    }
