import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from task_evaluation.vlm_json_parsing import parse_vlm_json_output_relaxed

DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
PROMPT_YAML_PATH = ROOT_DIR / "prompts" / "SemDesc_task_prompt.yaml"
FEW_SHOT_IMAGE_DIR = (
    ROOT_DIR / "dataset" / "MatPhaseBench-SemDesc" / "fewshot_images"
)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


from llm_service import (
    JIEKOU_BASE_URL as DEFAULT_JIEKOU_BASE_URL,
    TENCENT_BASE_URL as DEFAULT_TENCENT_BASE_URL,
    extract_vlm_output_text,
    extract_vlm_response_debug,
    is_empty_content_with_reasoning_json,
    is_finish_reason_length,
    parse_vlm_json_output as llm_service_parse_vlm_json_output,
    preview_json,
    preview_text,
    vlm_client,
)
from utils import load_prompt, read_json, save_to_json


def get_description_task_prompt(prompt_group, prompt_name):
    if not isinstance(prompt_group, dict):
        raise ValueError("specified_dimension_description_task must be a mapping.")

    prompt = prompt_group.get(prompt_name)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "Missing non-empty prompt at "
            f"specified_dimension_description_task.{prompt_name}"
        )

    return prompt


def get_description_task_prompt_group(prompt_group, prompt_name):
    if not isinstance(prompt_group, dict):
        raise ValueError("specified_dimension_description_task must be a mapping.")

    sub_prompts = prompt_group.get(prompt_name)
    if not isinstance(sub_prompts, dict) or not sub_prompts:
        raise ValueError(
            "Missing non-empty prompt group at "
            f"specified_dimension_description_task.{prompt_name}"
        )

    for sub_prompt_name, prompt in sub_prompts.items():
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "Missing non-empty prompt at "
                f"specified_dimension_description_task.{prompt_name}."
                f"{sub_prompt_name}"
            )

    return sub_prompts


DESCRIPTION_TASK_PROMPTS = load_prompt(
    PROMPT_YAML_PATH,
    "specified_dimension_description_task",
)
specified_dimension_description_prompt = get_description_task_prompt(
    DESCRIPTION_TASK_PROMPTS,
    "standard_description_task",
)
unlimited_description_prompt = get_description_task_prompt(
    DESCRIPTION_TASK_PROMPTS,
    "unlimited_description_task",
)
few_shot_description_prompts = get_description_task_prompt_group(
    DESCRIPTION_TASK_PROMPTS,
    "few_shot_description_task",
)
FEW_SHOT_DESCRIPTION_MODES = {
    "one_shot_description": {
        "prompt": few_shot_description_prompts["one_shot_description_task"],
        "shot_count": 1,
    },
    "three_shot_description": {
        "prompt": few_shot_description_prompts["three_shot_description_task"],
        "shot_count": 3,
    },
}
DIMENSION_DEFINITIONS = load_prompt(PROMPT_YAML_PATH, "DIMENSION_DEFINITIONS")


@dataclass(frozen=True)
class RuntimeConfig:
    input_paths: tuple[Path, ...]
    output_path: Path
    partial_output_path: Path
    failure_output_dir: Path
    img_base_dir: Path
    few_shot_image_dir: Path
    backend: str
    local_model: str
    local_base_url: str
    local_api_key: str
    dashscope_base_url: str
    model: str
    dashscope_api_key: str
    jiekou_base_url: str
    jiekou_api_key: str
    zhipu_base_url: str
    zhipu_api_key: str
    tencent_base_url: str
    tencent_api_key: str
    enable_thinking: bool
    temperature: float
    top_p: float
    max_tokens: int
    timeout_seconds: int
    retry_count: int
    retry_sleep_seconds: float
    save_interval: int
    max_workers: int
    reprocess_abnormal: bool
    description_mode: str
    basic_config: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use a VLM to describe phase diagram images from selected "
            "dimension_multi_classification labels."
        )
    )
    default_model = "qwen3.6-27b"
    default_enable_thinking = True
    parser.add_argument(
        "--img-base-dir",
        default=str(
            ROOT_DIR / "dataset" / "MatPhaseBench-SemDesc" / "phase_images"
        ),
        help="Path to the image folder.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to the VLM description output JSON file. If omitted, the "
            "file name is generated from --model and --enable-thinking."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "model_output"),
        help="Directory used for the generated output file when --output is omitted.",
    )
    parser.add_argument(
        "--failure-output-dir",
        default=str(ROOT_DIR / "phase_diagram_vlm_description_failures"),
        help="Directory for failed VLM responses.",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "dashscope", "jiekou", "zhipu", "tencent"),
        default="dashscope",
        help="VLM backend.",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Model name sent to the selected backend.",
    )
    parser.add_argument(
        "--local-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible local vLLM base URL.",
    )
    parser.add_argument(
        "--local-api-key",
        default="EMPTY",
        help="API key for the local OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--dashscope-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="DashScope OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--dashscope-api-key",
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="DashScope API key. Defaults to the DASHSCOPE_API_KEY environment variable.",
    )
    parser.add_argument(
        "--jiekou-base-url",
        default=DEFAULT_JIEKOU_BASE_URL,
        help="Jiekou OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--jiekou-api-key",
        default=os.getenv("JIEKOU_API_KEY", ""),
        help="Jiekou API key. Defaults to the JIEKOU_API_KEY environment variable.",
    )
    parser.add_argument(
        "--zhipu-base-url",
        default=DEFAULT_ZHIPU_BASE_URL,
        help="Zhipu OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--zhipu-api-key",
        default=os.getenv("ZHIPU_API_KEY", ""),
        help="Zhipu API key. Defaults to the ZHIPU_API_KEY environment variable.",
    )
    parser.add_argument(
        "--tencent-base-url",
        default=DEFAULT_TENCENT_BASE_URL,
        help="Tencent OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--tencent-api-key",
        default=os.getenv("TENCENT_API_KEY", ""),
        help="Tencent API key. Defaults to the TENCENT_API_KEY environment variable.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=default_enable_thinking,
        help="Enable model thinking/reasoning mode when supported.",
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=int, default=1000)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--description-mode", choices=("specified_dimension", "unlimited_description", "one_shot_description", "three_shot_description"), default="specified_dimension", help="Prompt/output mode. specified_dimension uses selected semantic dimensions and parses JSON; unlimited_description uses the unlimited prompt and parses diagram_description JSON; one_shot_description and three_shot_description use selected dimensions plus one or three few-shot images, respectively.")
    parser.add_argument(
        "--reprocess-abnormal",
        action="store_true",
        help=(
            "Reprocess records whose previous output has "
            "'vlm_description': 'Abnormal'. The full previous output list is "
            "preserved, and only Abnormal records are replaced."
        ),
    )
    parser.add_argument(
        "--local-model",
        default="",
        help=(
            "Model name used by the local OpenAI-compatible endpoint. "
            "Required when --backend local is selected."
        ),
    )
    return parser.parse_args()


def build_runtime_config(args):

    if args.backend != "local" and args.local_model:
        raise ValueError(
            f"--backend {args.backend} cannot be used together with --local-model. "
            "Remove --local-model or set --backend local."
        )
    if args.backend == "local" and not args.local_model:
        raise ValueError("--backend local requires a non-empty --local-model.")

    output_path = Path(args.output) if args.output else None
    output_model_name = args.local_model if args.backend == "local" else args.model
    if output_path is None:
        output_path = (
            Path(args.output_dir)
            / (
                f"{args.description_mode}_model_{output_model_name}"
                f"_think_{args.enable_thinking}"
                "_with_ground_truth.json"
            )
        )

    basic_config = {
        "input": [str(input_path) for input_path in args.input],
        "output_dir": str(args.output_dir),
        "backend": args.backend,
        "model": args.model,
        "local_model": args.local_model,
        "max_tokens": args.max_tokens,
        "max_workers": args.max_workers,
        "reprocess_abnormal": args.reprocess_abnormal,
        "description_mode": args.description_mode,
        "few_shot_image_dir": str(FEW_SHOT_IMAGE_DIR),
    }

    return RuntimeConfig(
        input_paths=tuple(Path(input_path) for input_path in args.input),
        output_path=output_path,
        partial_output_path=output_path.with_name(f"{output_path.stem}.partial.json"),
        failure_output_dir=Path(args.failure_output_dir),
        img_base_dir=Path(args.img_base_dir),
        few_shot_image_dir=FEW_SHOT_IMAGE_DIR,
        backend=args.backend,
        local_model=args.local_model,
        local_base_url=args.local_base_url,
        local_api_key=args.local_api_key or "EMPTY",
        dashscope_base_url=args.dashscope_base_url,
        model=args.model,
        dashscope_api_key=args.dashscope_api_key,
        jiekou_base_url=args.jiekou_base_url,
        jiekou_api_key=args.jiekou_api_key,
        zhipu_base_url=args.zhipu_base_url,
        zhipu_api_key=args.zhipu_api_key,
        tencent_base_url=args.tencent_base_url,
        tencent_api_key=args.tencent_api_key,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        retry_count=args.retry_count,
        retry_sleep_seconds=args.retry_sleep_seconds,
        save_interval=args.save_interval,
        max_workers=args.max_workers,
        reprocess_abnormal=args.reprocess_abnormal,
        description_mode=args.description_mode,
        basic_config=basic_config,
    )


def print_runtime_config(config):
    print(
        "Runtime config: "
        f"input={list(config.input_paths)} | "
        f"output={config.output_path} | "
        f"backend={config.backend} | "
        f"model={config.model} | "
        f"local_model={config.local_model} | "
        f"max_tokens={config.max_tokens} | "
        f"max_workers={config.max_workers} | "
        f"description_mode={config.description_mode} | "
        f"reprocess_abnormal={config.reprocess_abnormal}",
        flush=True,
    )


def build_aspect_definitions_text(selected_dimensions):
    definition_blocks = []

    for aspect in selected_dimensions:
        definition_blocks.append(f"- {aspect}:\n  {DIMENSION_DEFINITIONS[aspect]}")

    return "\n\n".join(definition_blocks)


def build_all_dimension_definitions_text():
    return build_aspect_definitions_text(DIMENSION_DEFINITIONS)


def build_dimension_description_prompt(
    prompt_template,
    selected_dimensions,
    use_all_dimension_definitions=False,
):
    def build_selected_dimensions_text(selected_dimensions):
        return "\n".join(f"- {aspect}" for aspect in selected_dimensions)

    def build_output_schema(selected_dimensions):
        output_schema = {
            "selected_dimensions": selected_dimensions,
            "descriptions": {
                aspect: {
                    "description": "<string>"
                }
                for aspect in selected_dimensions
            },
            "comprehensive_description": "<string>",
        }
        return json.dumps(output_schema, ensure_ascii=False, indent=2)

    if not selected_dimensions:
        raise ValueError("selected_dimensions cannot be empty.")

    invalid_aspects = set(selected_dimensions) - set(DIMENSION_DEFINITIONS)
    if invalid_aspects:
        raise ValueError(f"Invalid aspect labels: {sorted(invalid_aspects)}")

    dimension_definitions_text = build_aspect_definitions_text(selected_dimensions)
    if use_all_dimension_definitions:
        dimension_definitions_text = build_all_dimension_definitions_text()

    return prompt_template.format(
        SELECTED_DIMENSIONS=build_selected_dimensions_text(selected_dimensions),
        DIMENSION_DEFINITIONS=dimension_definitions_text,
        OUTPUT_SCHEMA=build_output_schema(selected_dimensions),
    )


def parse_unlimited_description_result(raw_text):
    parsed_result = parse_vlm_json_output_relaxed(
        raw_text,
        strict_parser=llm_service_parse_vlm_json_output,
    )
    if not isinstance(parsed_result, dict):
        raise ValueError("unlimited_description output must be a JSON object.")

    diagram_description = parsed_result.get("diagram_description")
    if not isinstance(diagram_description, str) or not diagram_description.strip():
        raise ValueError(
            "unlimited_description output must contain a non-empty "
            "diagram_description string."
        )

    return {"diagram_description": diagram_description.strip()}


def get_selected_dimensions(item):
    associative_matching_result = item.get("associative_matching_result")
    classification = (
        associative_matching_result.get("dimension_multi_classification")
        if isinstance(associative_matching_result, dict)
        else None
    )
    if not isinstance(classification, dict):
        return []

    labels = classification.get("labels")
    if not isinstance(labels, list):
        return []

    return [
        label
        for label in labels
        if isinstance(label, str) and label in DIMENSION_DEFINITIONS
    ]


def build_image_path_from_base(image_path, config):
    image_path = image_path.strip()
    parsed = urlparse(image_path)
    if parsed.scheme in ("http", "https", "file"):
        return image_path

    path = Path(image_path)
    if path.is_absolute():
        try:
            path.relative_to(config.img_base_dir)
            return str(path)
        except ValueError:
            return str(config.img_base_dir / image_path.lstrip("/\\"))

    return str(config.img_base_dir / image_path.lstrip("/\\"))


def get_few_shot_image_paths(config, shot_count):
    if not config.few_shot_image_dir.is_dir():
        raise FileNotFoundError(
            f"Few-shot image directory not found: {config.few_shot_image_dir}"
        )

    image_paths = [
        path
        for path in sorted(
            config.few_shot_image_dir.iterdir(),
            key=lambda item: item.name.lower(),
        )
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    if len(image_paths) < shot_count:
        raise ValueError(
            f"Expected at least {shot_count} few-shot image(s), but found "
            f"{len(image_paths)} in: {config.few_shot_image_dir}"
        )

    return [str(path) for path in image_paths[:shot_count]]


def get_record_image_path(item):
    for image_key in ("img_path", "image_path"):
        image_path = item.get(image_key)
        if isinstance(image_path, str) and image_path.strip():
            return image_path

    sample_id = item.get("sample_id")
    if isinstance(sample_id, int):
        return f"images/phase_{sample_id:03d}.jpg"

    raise ValueError(
        "Each input record must contain a sample_id or a non-empty image_path/img_path field."
    )


def normalize_sample_id(sample_id):
    if isinstance(sample_id, bool):
        return str(sample_id)
    if isinstance(sample_id, int):
        return str(sample_id)
    if isinstance(sample_id, float) and sample_id.is_integer():
        return str(int(sample_id))
    return str(sample_id).strip()


def get_ground_truth_text(item):
    associative_matching_result = item.get("associative_matching_result")
    if not isinstance(associative_matching_result, dict):
        return None

    ground_truth = associative_matching_result.get("two_stage_llm_integration")
    if isinstance(ground_truth, str) and ground_truth.strip():
        return ground_truth

    return None


def build_ground_truth_index(source_items):
    input_index_to_ground_truth = {}
    missing_ground_truth_records = []

    for fallback_index, item in enumerate(source_items):
        input_index = item.get("_input_index")
        if not isinstance(input_index, int):
            input_index = fallback_index
        ground_truth = get_ground_truth_text(item)
        if ground_truth is None:
            missing_ground_truth_records.append(
                {
                    "_input_index": input_index,
                    "sample_id": item.get("sample_id"),
                }
            )
            continue

        input_index_to_ground_truth[input_index] = ground_truth

    if missing_ground_truth_records:
        raise ValueError(
            "Input records missing string "
            "associative_matching_result.two_stage_llm_integration: "
            f"{missing_ground_truth_records}"
        )

    return input_index_to_ground_truth


def add_ground_truth_to_record(record, ground_truth):
    updated_record = {}
    inserted = False

    for key, value in record.items():
        if key == "ground_truth":
            continue
        updated_record[key] = value
        if key == "vlm_description":
            updated_record["ground_truth"] = ground_truth
            inserted = True

    if not inserted:
        updated_record["ground_truth"] = ground_truth

    return updated_record


def add_ground_truth_to_results(results, source_items):
    ground_truth_index = build_ground_truth_index(source_items)
    enriched_results = []
    missing_ground_truth_records = []

    for fallback_index, record in enumerate(results):
        if not isinstance(record, dict):
            enriched_results.append(record)
            continue

        input_index = record.get("_input_index")
        if not isinstance(input_index, int):
            input_index = fallback_index
        ground_truth = ground_truth_index.get(input_index)
        if ground_truth is None:
            missing_ground_truth_records.append(
                {
                    "_input_index": input_index,
                    "sample_id": record.get("sample_id"),
                }
            )
            continue

        enriched_results.append(add_ground_truth_to_record(record, ground_truth))

    if missing_ground_truth_records:
        raise ValueError(
            "Output records are missing from input ground-truth index: "
            f"{missing_ground_truth_records}"
        )

    return enriched_results


def save_failure(sample_id, attempt, raw_text, raw_response, config, error=None):
    config.failure_output_dir.mkdir(parents=True, exist_ok=True)
    safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))
    output_path = config.failure_output_dir / (
        f"sample_{safe_sample_id}_attempt_{attempt}.json"
    )
    response_debug = extract_vlm_response_debug(raw_response)
    save_to_json(
        {
            "sample_id": sample_id,
            "attempt": attempt,
            "basic_config": config.basic_config,
            "error": str(error) if error else None,
            "raw_text_preview": preview_text(raw_text),
            "raw_text": raw_text,
            "finish_reason": response_debug.get("finish_reason"),
            "message_content_preview": preview_text(response_debug.get("content")),
            "reasoning_content_preview": preview_text(
                response_debug.get("reasoning_content")
            ),
            "reasoning_preview": preview_text(response_debug.get("reasoning")),
            "raw_response_preview": preview_json(raw_response),
            "raw_response": raw_response,
        },
        output_path,
        output_info=False,
    )
    return output_path


def build_abnormal_phase_description_result(
    sample_id,
    image_path,
    selected_dimensions,
    ground_truth,
    failure_path,
    abnormal_reason,
    source_fields=None,
):
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "selected_dimensions": selected_dimensions,
        "vlm_description": "Abnormal",
        "ground_truth": ground_truth,
        "abnormal_reason": abnormal_reason,
        "failure_path": str(failure_path.resolve()),
        **(source_fields or {}),
    }


def is_abnormal_result_record(record):
    if not isinstance(record, dict):
        return False

    vlm_description = record.get("vlm_description")
    return (
        vlm_description == "Abnormal"
        or (
            isinstance(vlm_description, dict)
            and "error" in vlm_description
        )
    )


def get_input_tracking_fields(item):
    fields = {}
    for field_name in (
        "_input_index",
        "_input_source_file",
        "_input_source_index",
    ):
        if isinstance(item, dict) and field_name in item:
            fields[field_name] = item[field_name]
    return fields


def get_processable_items(data):
    return [item for item in data if isinstance(item, dict)]


def load_input_data(input_paths):
    merged_data = []
    for input_path in input_paths:
        records = read_json(input_path)
        if not isinstance(records, list):
            raise ValueError(f"Input JSON file must contain a list: {input_path}")

        resolved_input_path = str(Path(input_path).resolve())
        for source_index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                merged_data.append(record)
                continue

            item = dict(record)
            item["_input_index"] = len(merged_data)
            item["_input_source_file"] = resolved_input_path
            item["_input_source_index"] = source_index
            merged_data.append(item)

    return merged_data


def save_json_atomic(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_output_path.replace(output_path)


def save_partial_results(results, config, source_items=None):
    if isinstance(results, dict):
        if source_items is None:
            partial_records = [results[index] for index in sorted(results)]
        else:
            partial_records = [
                results[index]
                for index in range(len(source_items))
                if index in results
            ]
    else:
        partial_records = results

    save_json_atomic(partial_records, config.partial_output_path)
    print(
        f"\nSaved partial VLM phase description result "
        f"({len(partial_records)} records) to {config.partial_output_path.resolve()}"
    )


def load_resume_results(source_items, config):
    if not config.partial_output_path.exists():
        return {}

    resume_results = read_json(config.partial_output_path)
    if len(resume_results) > len(source_items):
        raise ValueError(
            "Partial output has more records than input data: "
            f"{config.partial_output_path.resolve()}"
        )

    input_index_to_index = {
        item.get("_input_index"): index
        for index, item in enumerate(source_items)
        if isinstance(item, dict) and isinstance(item.get("_input_index"), int)
    }
    sample_id_to_index = {}
    duplicate_input_sample_ids = []
    for index, item in enumerate(source_items):
        sample_id = item.get("sample_id")
        if sample_id in sample_id_to_index:
            duplicate_input_sample_ids.append(sample_id)
        sample_id_to_index[sample_id] = index

    allow_sample_id_fallback = not duplicate_input_sample_ids

    resume_results_by_index = {}
    duplicate_sample_ids = []
    duplicate_input_indices = []

    for result_position, result in enumerate(resume_results):
        result_input_index = (
            result.get("_input_index") if isinstance(result, dict) else None
        )
        result_sample_id = result.get("sample_id") if isinstance(result, dict) else None

        if isinstance(result_input_index, int):
            if result_input_index not in input_index_to_index:
                raise ValueError(
                    "Partial output contains an _input_index missing from input "
                    f"data at partial_index={result_position}: "
                    f"_input_index={result_input_index}, "
                    f"file={config.partial_output_path.resolve()}"
                )
            source_index = input_index_to_index[result_input_index]
        elif allow_sample_id_fallback:
            if result_sample_id not in sample_id_to_index:
                raise ValueError(
                    "Partial output contains a sample_id missing from input data at "
                    f"partial_index={result_position}: "
                    f"partial sample_id={result_sample_id}, "
                    f"file={config.partial_output_path.resolve()}"
                )
            source_index = sample_id_to_index[result_sample_id]
        else:
            raise ValueError(
                "Partial output does not contain _input_index, and input data has "
                "duplicate sample_id values. Remove the stale partial output or "
                "rerun from scratch."
            )

        if source_index in resume_results_by_index:
            if isinstance(result_input_index, int):
                duplicate_input_indices.append(result_input_index)
            else:
                duplicate_sample_ids.append(result_sample_id)
        resume_results_by_index[source_index] = result

    if duplicate_input_indices:
        raise ValueError(
            "Partial output contains duplicate _input_index values: "
            f"{sorted(duplicate_input_indices)}"
        )
    if duplicate_sample_ids:
        raise ValueError(
            "Partial output contains duplicate sample_id values: "
            f"{sorted(duplicate_sample_ids)}"
        )

    print(f"Resuming from partial output: {config.partial_output_path.resolve()}")
    print(f"Skipping {len(resume_results_by_index)} already processed records.")
    return resume_results_by_index


def remove_partial_output(config):
    if config.partial_output_path.exists():
        config.partial_output_path.unlink()
        print(f"Removed partial output: {config.partial_output_path.resolve()}")


def describe_phase_diagram(item, config):
    sample_id = item.get("sample_id")
    source_fields = get_input_tracking_fields(item)
    record_image_path = get_record_image_path(item)
    ground_truth = get_ground_truth_text(item)
    image_path = build_image_path_from_base(record_image_path, config)

    if config.description_mode == "specified_dimension":
        selected_dimensions = get_selected_dimensions(item)
        prompt = build_dimension_description_prompt(
            specified_dimension_description_prompt,
            selected_dimensions,
        )
        vlm_image_input = image_path
    elif config.description_mode == "unlimited_description":
        selected_dimensions = None
        prompt = unlimited_description_prompt
        vlm_image_input = image_path
    elif config.description_mode in FEW_SHOT_DESCRIPTION_MODES:
        few_shot_mode = FEW_SHOT_DESCRIPTION_MODES[config.description_mode]
        selected_dimensions = get_selected_dimensions(item)
        prompt = build_dimension_description_prompt(
            few_shot_mode["prompt"],
            selected_dimensions,
            use_all_dimension_definitions=True,
        )
        vlm_image_input = get_few_shot_image_paths(
            config,
            few_shot_mode["shot_count"],
        ) + [image_path]
    else:
        raise ValueError(
            "Unsupported description mode. Use 'specified_dimension' or "
            "'unlimited_description' or 'one_shot_description' or "
            "'three_shot_description'."
        )

    if config.backend == "local":
        model = config.local_model
        base_url = config.local_base_url
        api_key = config.local_api_key or "EMPTY"
    elif config.backend == "dashscope":
        model = config.model
        base_url = config.dashscope_base_url
        api_key = config.dashscope_api_key
    elif config.backend == "jiekou":
        model = config.model
        base_url = config.jiekou_base_url
        api_key = config.jiekou_api_key
    elif config.backend == "zhipu":
        model = config.model
        base_url = config.zhipu_base_url
        api_key = config.zhipu_api_key
    elif config.backend == "tencent":
        model = config.model
        base_url = config.tencent_base_url
        api_key = config.tencent_api_key
    else:
        raise ValueError(
            "Unsupported backend. Use 'local', 'dashscope', 'jiekou', 'zhipu', "
            "or 'tencent'."
        )

    last_error = None
    last_raw_text = ""
    last_raw_response = None
    failure_paths = []
    for attempt in range(1, config.retry_count + 1):
        raw_text = ""
        raw_response = None
        try:
            # print("[Debug] prompt = ", prompt)
            response = vlm_client(
                prompt=prompt,
                backend=config.backend,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                enable_thinking=config.enable_thinking,
                timeout=config.timeout_seconds,
                top_k=20,
                image=vlm_image_input,
            )
            raw_text = response.get("raw_text", "") if isinstance(response, dict) else ""
            raw_response = (
                response.get("raw_response") if isinstance(response, dict) else None
            )
            if not raw_text and raw_response:
                raw_text = extract_vlm_output_text(raw_response)

            if isinstance(response, dict) and response.get("success") is not True:
                if not raw_text and not raw_response:
                    raise RuntimeError(
                        "VLM service call failed: "
                        f"backend={response.get('backend')} | "
                        f"model={response.get('model')} | "
                        f"error={response.get('error')}"
                    )

            raw_text = raw_text.strip()

            last_raw_text = raw_text
            last_raw_response = raw_response
            if config.description_mode == "unlimited_description":
                if not raw_text:
                    raise ValueError("VLM output is empty.")
                parsed_result = parse_unlimited_description_result(raw_text)

                return {
                    "sample_id": sample_id,
                    "image_path": record_image_path,
                    "selected_dimensions": None,
                    "vlm_description": parsed_result,
                    "ground_truth": ground_truth,
                    **source_fields,
                }

            parsed_result = parse_vlm_json_output_relaxed(
                raw_text,
                strict_parser=llm_service_parse_vlm_json_output,
            )
            # print("[Debug] parsed_result = ", parsed_result)

            return {
                "sample_id": sample_id,
                "image_path": record_image_path,
                "selected_dimensions": selected_dimensions,
                "vlm_description": parsed_result,
                "ground_truth": ground_truth,
                **source_fields,
            }
        except Exception as error:
            last_error = error
            last_raw_text = raw_text
            last_raw_response = raw_response
            failure_path = save_failure(
                sample_id,
                attempt,
                raw_text,
                raw_response,
                config,
                error=error,
            )
            failure_paths.append(failure_path)

            if is_empty_content_with_reasoning_json(raw_text, raw_response):
                return build_abnormal_phase_description_result(
                    sample_id=sample_id,
                    image_path=record_image_path,
                    selected_dimensions=selected_dimensions,
                    ground_truth=ground_truth,
                    failure_path=failure_path,
                    abnormal_reason=(
                        "message.content is empty while reasoning_content contains JSON. "
                        "Only message.content is accepted as the final answer."
                    ),
                    source_fields=source_fields,
                )

            if is_finish_reason_length(raw_response):
                return build_abnormal_phase_description_result(
                    sample_id=sample_id,
                    image_path=record_image_path,
                    selected_dimensions=selected_dimensions,
                    ground_truth=ground_truth,
                    failure_path=failure_path,
                    abnormal_reason=(
                        "finish_reason is length, indicating the model output was "
                        "truncated before a valid JSON answer was available."
                    ),
                    source_fields=source_fields,
                )

            if attempt < config.retry_count:
                time.sleep(config.retry_sleep_seconds)

    response_debug = extract_vlm_response_debug(last_raw_response)
    raise RuntimeError(
        f"VLM phase diagram description failed for sample_id={sample_id}: {last_error}\n"
        f"Last raw output preview:\n{preview_text(last_raw_text)}\n"
        f"Last finish_reason: {response_debug.get('finish_reason')}\n"
        f"Last content preview:\n{preview_text(response_debug.get('content'))}\n"
        f"Last reasoning_content preview:\n{preview_text(response_debug.get('reasoning_content'))}\n"
        "Saved failed VLM responses:\n"
        + "\n".join(str(path.resolve()) for path in failure_paths)
    )


def process_data(data, config):
    source_items = get_processable_items(data)
    completed_results = load_resume_results(source_items, config)
    pending_items = [
        (index, item)
        for index, item in enumerate(source_items)
        if index not in completed_results
    ]

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_index = {
            executor.submit(describe_phase_diagram, item, config): index
            for index, item in pending_items
        }

        with tqdm(
            total=len(source_items),
            initial=len(completed_results),
            desc="Describing phase diagrams",
            unit="item",
        ) as progress_bar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    completed_results[index] = future.result()
                except Exception:
                    for completed_future, completed_index in future_to_index.items():
                        if (
                            completed_future is future
                            or completed_index in completed_results
                            or not completed_future.done()
                            or completed_future.cancelled()
                        ):
                            continue
                        try:
                            completed_results[completed_index] = completed_future.result()
                        except Exception:
                            pass

                    for pending_future in future_to_index:
                        pending_future.cancel()
                    save_partial_results(completed_results, config, source_items)
                    raise

                progress_bar.update(1)

                if len(completed_results) % config.save_interval == 0:
                    save_partial_results(completed_results, config, source_items)

    if completed_results and len(completed_results) % config.save_interval != 0:
        save_partial_results(completed_results, config, source_items)

    missing_indices = [
        index
        for index in range(len(source_items))
        if index not in completed_results
    ]
    if missing_indices:
        missing_sample_ids = [
            source_items[index].get("sample_id")
            for index in missing_indices
        ]
        raise RuntimeError(
            "Some samples were not processed: "
            f"indices={missing_indices}, sample_ids={missing_sample_ids}"
        )

    return [
        completed_results[index]
        for index in range(len(source_items))
    ]


def load_error_reprocessing_base_results(config):
    if config.partial_output_path.exists():
        try:
            partial_results = read_json(config.partial_output_path)
            print(
                "Resuming error reprocessing from partial output: "
                f"{config.partial_output_path.resolve()}"
            )
            return partial_results
        except json.JSONDecodeError as error:
            if not config.output_path.exists():
                raise
            print(
                "Ignoring invalid partial output and falling back to previous "
                f"final output: partial={config.partial_output_path.resolve()} | "
                f"error={error}"
            )

    if not config.output_path.exists():
        raise FileNotFoundError(
            "Cannot reprocess Abnormal records because previous output does not exist: "
            f"{config.output_path.resolve()}"
        )

    initial_results = read_json(config.output_path)
    print(
        "Loaded previous output for Abnormal reprocessing: "
        f"{config.output_path.resolve()}"
    )
    return initial_results


def process_error_reprocessing(data, config):
    source_items = get_processable_items(data)
    input_index_to_item = {
        item.get("_input_index"): item
        for item in source_items
        if isinstance(item.get("_input_index"), int)
    }
    sample_id_to_item = {}
    duplicate_input_sample_ids = []
    for item in source_items:
        sample_id = item.get("sample_id")
        if sample_id in sample_id_to_item:
            duplicate_input_sample_ids.append(sample_id)
        sample_id_to_item[sample_id] = item

    allow_sample_id_fallback = not duplicate_input_sample_ids

    results = load_error_reprocessing_base_results(config)

    targets = []
    missing_records = []
    for result_index, result in enumerate(results):
        if not is_abnormal_result_record(result):
            continue

        result_input_index = result.get("_input_index")
        sample_id = result.get("sample_id")
        if isinstance(result_input_index, int):
            source_item = input_index_to_item.get(result_input_index)
        elif allow_sample_id_fallback:
            source_item = sample_id_to_item.get(sample_id)
        else:
            raise ValueError(
                "Previous output does not contain _input_index, and input data has "
                "duplicate sample_id values. Remove the stale previous output or "
                "rerun from scratch."
            )

        if source_item is None:
            missing_records.append(
                {
                    "result_index": result_index,
                    "_input_index": result_input_index,
                    "sample_id": sample_id,
                }
            )
            continue
        targets.append((result_index, source_item))

    if missing_records:
        raise ValueError(
            "Previous output contains abnormal/error records missing from INPUT: "
            f"{missing_records}"
        )

    print(f"Found {len(targets)} abnormal/error records to reprocess.")
    if not targets:
        return results

    processed_count = 0
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_index = {
            executor.submit(describe_phase_diagram, source_item, config): index
            for index, source_item in targets
        }

        with tqdm(
            total=len(targets),
            desc="Reprocessing Abnormal phase diagrams",
            unit="item",
        ) as progress_bar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception:
                    for pending_future in future_to_index:
                        pending_future.cancel()
                    save_partial_results(results, config)
                    raise

                processed_count += 1
                progress_bar.update(1)
                if processed_count % config.save_interval == 0:
                    save_partial_results(results, config)

    if processed_count % config.save_interval != 0:
        save_partial_results(results, config)

    return results


def main():
    args = parse_args()
    config = build_runtime_config(args)
    print_runtime_config(config)

    data = load_input_data(config.input_paths)
    if config.reprocess_abnormal:
        results = process_error_reprocessing(data, config)
    else:
        results = process_data(data, config)

    results = add_ground_truth_to_results(results, get_processable_items(data))
    save_to_json(results, config.output_path, output_info=False)
    print(f"Wrote VLM phase description result to {config.output_path.resolve()}")
    remove_partial_output(config)


if __name__ == "__main__":
    main()
