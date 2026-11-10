from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm_service import parse_llm_json_output, preview_text, vlm_client
from src.utils import load_prompt


JUDGE_PROMPT_PATH = ROOT_DIR / "prompts" / "semantic_quality_score.yaml"
SEMDESC_PROMPT_PATH = ROOT_DIR / "prompts" / "SemDesc_task_prompt.yaml"
JUDGE_PROMPT_GROUP = "llm_as_judge_semantic_quality_score"
SCORE_FIELDS_BY_SECTION = {
    "semantic_completeness": (
        "semantic_dimension_completeness",
        "basic_information_completeness",
        "reasoning_description_completeness",
    ),
    "information_consistency": (
        "semantic_dimension_consistency",
        "terminology_consistency",
        "entity_relationship_consistency",
    ),
    "visual_content_alignment": (
        "visual_entity_alignment",
        "spatial_relationship_alignment",
        "numerical_relationship_alignment",
    ),
}
SECTION_BY_DIMENSION = {
    dimension_name: section_name
    for section_name, dimension_names in SCORE_FIELDS_BY_SECTION.items()
    for dimension_name in dimension_names
}
VISUAL_DIMENSIONS = set(SCORE_FIELDS_BY_SECTION["visual_content_alignment"])
# Dimensions frozen after reaching the Gwet's AC1 target.
METRIC_COMPLETED_FROZEN_DIMENSION_SCORES = {
}
# Dimensions temporarily frozen only to isolate the completed-dimension tests.
TARGETED_TEST_TEMPORARY_FROZEN_DIMENSION_SCORES = {
}
# Calibration-only shortcut: clear both source mappings before production runs.
CALIBRATION_FROZEN_DIMENSION_SCORES = {
    **METRIC_COMPLETED_FROZEN_DIMENSION_SCORES,
    **TARGETED_TEST_TEMPORARY_FROZEN_DIMENSION_SCORES,
}


def _frozen_dimension_reason(dimension_name: str) -> str:
    if dimension_name in METRIC_COMPLETED_FROZEN_DIMENSION_SCORES:
        return "Metric-target-completed frozen score; no LLM call was made."
    return "Targeted-test temporary frozen score; no LLM call was made."


class LLMAsJudgeCallError(RuntimeError):
    def __init__(self, message: str, response: dict[str, Any] | None = None):
        super().__init__(message)
        self.response = response


def _ensure_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _load_judge_prompts() -> tuple[dict[str, Any], str | None]:
    prompt_group = load_prompt(JUDGE_PROMPT_PATH, JUDGE_PROMPT_GROUP)
    if not isinstance(prompt_group, dict):
        raise ValueError(f"{JUDGE_PROMPT_GROUP} must be a mapping.")

    system_prompt = prompt_group.get("system")
    if system_prompt is not None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(
                "Shared system prompt must be a non-empty string when provided."
            )
        system_prompt = system_prompt.strip()

    return prompt_group, system_prompt


def _build_judge_prompt(
    prompt_group: dict[str, Any],
    section_name: str,
    dimension_name: str,
    **placeholders: str,
) -> str:
    section = prompt_group.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(
            f"Missing prompt section at {JUDGE_PROMPT_GROUP}.{section_name}."
        )
    template = section.get(dimension_name)
    if not isinstance(template, str) or not template.strip():
        raise ValueError(
            f"Missing non-empty prompt at "
            f"{JUDGE_PROMPT_GROUP}.{section_name}.{dimension_name}."
        )
    return template.format(**placeholders)


def _call_judge_llm(
    prompt: str,
    *,
    image: str | Sequence[str] | None = None,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    top_p: float = 0.8,
    max_tokens: int = 16384,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    response = vlm_client(
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
        presence_penalty=presence_penalty,
        top_k=top_k,
        system_prompt=system_prompt,
        image=image,
    )
    if not isinstance(response, dict):
        raise LLMAsJudgeCallError(
            f"LLM-as-Judge response is not a dict: {type(response)}"
        )

    if response.get("success") is True:
        answer = response.get("answer")
        if isinstance(answer, dict):
            return answer
        raise RuntimeError(f"LLM-as-Judge answer is not a dict: {type(answer)}")

    raw_text = response.get("raw_text")
    parse_error = None
    if isinstance(raw_text, str) and raw_text.strip():
        try:
            parsed_answer = parse_llm_json_output(raw_text)
            if isinstance(parsed_answer, dict):
                return parsed_answer
        except Exception as error:
            parse_error = error

    error_message = (
        "LLM-as-Judge call failed: "
        f"backend={response.get('backend')} | "
        f"model={response.get('model')} | "
        f"error={response.get('error')} | "
        f"raw_text={preview_text(raw_text, limit=1000)}"
    )
    if parse_error is not None:
        error_message += f" | raw_text_parse_error={parse_error}"
    raise LLMAsJudgeCallError(error_message, response=response)


def _calculate_total_score(
    judge_answer: dict[str, Any],
    section_name: str,
) -> int | float:
    section = judge_answer.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"LLM-as-Judge output missing section: {section_name}")

    scores = []
    for score_field in SCORE_FIELDS_BY_SECTION[section_name]:
        score_item = section.get(score_field)
        if not isinstance(score_item, dict):
            raise ValueError(
                f"LLM-as-Judge output missing score object: "
                f"{section_name}.{score_field}"
            )

        score = score_item.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(
                f"LLM-as-Judge score must be numeric: "
                f"{section_name}.{score_field}.score"
            )
        if not float(score).is_integer():
            raise ValueError(
                f"LLM-as-Judge score must be an integer: "
                f"{section_name}.{score_field}.score={score}"
            )
        score = int(score)
        if score < 0 or score > 2:
            raise ValueError(
                f"LLM-as-Judge score out of range [0, 2]: "
                f"{section_name}.{score_field}.score={score}"
            )
        scores.append(score)

    # The three dimension scores are authoritative; the section score is
    # recomputed rather than trusted so a miscounted echo cannot discard them.
    total_score = sum(scores)
    section["score"] = total_score
    return total_score


def _evaluate_dimension(
    prompt_group: dict[str, Any],
    shared_system_prompt: str | None,
    dimension_name: str,
    *,
    ground_truth: str = "",
    evaluated_text: str,
    selected_dimensions: str = "",
    dimension_definitions: str = "",
    image: str | Sequence[str] | None = None,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    top_p: float = 0.8,
    max_tokens: int = 16384,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    frozen_score = CALIBRATION_FROZEN_DIMENSION_SCORES.get(dimension_name)
    if frozen_score is not None:
        return {
            "score": frozen_score,
            "reason": _frozen_dimension_reason(dimension_name),
        }

    prompt = _build_judge_prompt(
        prompt_group,
        SECTION_BY_DIMENSION[dimension_name],
        dimension_name,
        GROUND_TRUTH=ground_truth,
        SELECTED_DIMENSIONS=selected_dimensions,
        DIMENSION_DEFINITIONS=dimension_definitions,
        EVALUATED_TEXT=evaluated_text,
    )
    try:
        judge_answer = _call_judge_llm(
            prompt,
            image=image if dimension_name in VISUAL_DIMENSIONS else None,
            backend=backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
            presence_penalty=presence_penalty,
            top_k=top_k,
            system_prompt=(
                system_prompt if system_prompt is not None else shared_system_prompt
            ),
        )
    except LLMAsJudgeCallError as error:
        raise LLMAsJudgeCallError(
            f"LLM-as-Judge dimension call failed: {dimension_name} | {error}",
            response=error.response,
        ) from error

    dimension_result = judge_answer.get(dimension_name, judge_answer)
    if not isinstance(dimension_result, dict):
        raise ValueError(
            f"LLM-as-Judge output for {dimension_name} must be an object."
        )
    score = dimension_result.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(f"LLM-as-Judge score must be numeric: {dimension_name}.score")
    if not float(score).is_integer() or int(score) not in (0, 1, 2):
        raise ValueError(
            f"LLM-as-Judge score must be an integer in [0, 2]: "
            f"{dimension_name}.score={score}"
        )
    reason = dimension_result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            f"LLM-as-Judge reason must be non-empty: {dimension_name}.reason"
        )
    return {"score": int(score), "reason": reason.strip()}


def semantic_completeness_score_calculation(
    ground_truth: str,
    evaluated_text: str,
    selected_dimensions: Sequence[str],
    *,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    top_p: float = 0.8,
    max_tokens: int = 16384,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any], int | float]:
    dimension_definitions = load_prompt(SEMDESC_PROMPT_PATH, "DIMENSION_DEFINITIONS")
    if not isinstance(dimension_definitions, dict):
        raise ValueError("DIMENSION_DEFINITIONS must be a mapping.")

    if isinstance(selected_dimensions, (str, bytes)) or not selected_dimensions:
        raise ValueError("selected_dimensions must be a non-empty sequence.")

    selected_dimension_count = len(selected_dimensions)
    selected_dimensions = [
        dimension.strip()
        for dimension in selected_dimensions
        if isinstance(dimension, str) and dimension.strip()
    ]
    if len(selected_dimensions) != selected_dimension_count:
        raise ValueError("selected_dimensions entries must be non-empty strings.")

    invalid_dimensions = set(selected_dimensions) - set(dimension_definitions)
    if invalid_dimensions:
        raise ValueError(f"Invalid selected_dimensions: {sorted(invalid_dimensions)}")

    dimension_definitions_text = "\n\n".join(
        f"- {dimension}:\n  {dimension_definitions[dimension]}"
        for dimension in selected_dimensions
    )
    prompt_group, shared_system_prompt = _load_judge_prompts()
    ground_truth = _ensure_text(ground_truth, "ground_truth")
    evaluated_text = _ensure_text(evaluated_text, "evaluated_text")
    selected_dimensions_text = "\n".join(
        f"- {dimension}" for dimension in selected_dimensions
    )
    dimension_results = {}
    for dimension_name in SCORE_FIELDS_BY_SECTION["semantic_completeness"]:
        frozen_score = CALIBRATION_FROZEN_DIMENSION_SCORES.get(dimension_name)
        if frozen_score is not None:
            dimension_results[dimension_name] = {
                "score": frozen_score,
                "reason": _frozen_dimension_reason(dimension_name),
            }
            continue
        dimension_results[dimension_name] = _evaluate_dimension(
            prompt_group,
            shared_system_prompt,
            dimension_name,
            ground_truth=ground_truth,
            evaluated_text=evaluated_text,
            selected_dimensions=selected_dimensions_text,
            dimension_definitions=dimension_definitions_text,
            backend=backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
            presence_penalty=presence_penalty,
            top_k=top_k,
            system_prompt=system_prompt,
        )
    judge_answer = {
        "semantic_completeness": {
            **dimension_results,
            "reason": "Aggregated from independent dimension evaluations.",
        }
    }
    return judge_answer, _calculate_total_score(
        judge_answer,
        "semantic_completeness",
    )


def information_consistency_score_calculation(
    ground_truth: str,
    evaluated_text: str,
    *,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    top_p: float = 0.8,
    max_tokens: int = 16384,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any], int | float]:
    prompt_group, shared_system_prompt = _load_judge_prompts()
    ground_truth = _ensure_text(ground_truth, "ground_truth")
    evaluated_text = _ensure_text(evaluated_text, "evaluated_text")
    dimension_results = {
        dimension_name: _evaluate_dimension(
            prompt_group,
            shared_system_prompt,
            dimension_name,
            ground_truth=ground_truth,
            evaluated_text=evaluated_text,
            backend=backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
            presence_penalty=presence_penalty,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        for dimension_name in SCORE_FIELDS_BY_SECTION["information_consistency"]
    }
    judge_answer = {
        "information_consistency": {
            **dimension_results,
            "reason": "Aggregated from independent dimension evaluations.",
        }
    }
    return judge_answer, _calculate_total_score(
        judge_answer,
        "information_consistency",
    )


def visual_content_alignment_score_calculation(
    evaluated_text: str,
    image: str | Sequence[str],
    *,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    top_p: float = 0.8,
    max_tokens: int = 16384,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any], int | float]:
    prompt_group, shared_system_prompt = _load_judge_prompts()
    evaluated_text = _ensure_text(evaluated_text, "evaluated_text")
    dimension_results = {
        dimension_name: _evaluate_dimension(
            prompt_group,
            shared_system_prompt,
            dimension_name,
            evaluated_text=evaluated_text,
            image=image,
            backend=backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
            presence_penalty=presence_penalty,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        for dimension_name in SCORE_FIELDS_BY_SECTION["visual_content_alignment"]
    }
    judge_answer = {
        "visual_content_alignment": {
            **dimension_results,
            "reason": "Aggregated from independent dimension evaluations.",
        }
    }
    return judge_answer, _calculate_total_score(
        judge_answer,
        "visual_content_alignment",
    )


def llm_as_judge_description_quality_assessment(
    sample: dict[str, Any],
    *,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    top_p: float = 0.8,
    max_tokens: int = 16384,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise TypeError(f"sample must be a dict: {type(sample)}")

    sample_id = sample.get("sample_id")
    ground_truth = sample.get("ground_truth")
    vlm_description = sample.get("vlm_description")
    if not isinstance(vlm_description, dict):
        raise ValueError("sample.vlm_description must be a dict.")
    evaluated_text = vlm_description.get("comprehensive_description")
    selected_dimensions = sample.get("selected_dimensions")
    image = sample.get("image_path")
    if not isinstance(image, str) or not image.strip():
        raise ValueError("sample.image_path must be a non-empty string.")

    llm_kwargs = {
        "backend": backend,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "timeout": timeout,
        "presence_penalty": presence_penalty,
        "top_k": top_k,
        "system_prompt": system_prompt,
    }
    semantic_result, semantic_total_score = semantic_completeness_score_calculation(
        ground_truth=ground_truth,
        evaluated_text=evaluated_text,
        selected_dimensions=selected_dimensions,
        **llm_kwargs,
    )
    consistency_result, consistency_total_score = (
        information_consistency_score_calculation(
            ground_truth=ground_truth,
            evaluated_text=evaluated_text,
            **llm_kwargs,
        )
    )
    visual_result, visual_total_score = visual_content_alignment_score_calculation(
        evaluated_text=evaluated_text,
        image=image,
        **llm_kwargs,
    )

    return {
        "sample_id": sample_id,
        "semantic_completeness": semantic_result,
        "semantic_completeness_total_score": semantic_total_score,
        "information_consistency": consistency_result,
        "information_consistency_total_score": consistency_total_score,
        "visual_content_alignment": visual_result,
        "visual_content_alignment_total_score": visual_total_score,
        "total_score": (
            semantic_total_score
            + consistency_total_score
            + visual_total_score
        ),
    }
