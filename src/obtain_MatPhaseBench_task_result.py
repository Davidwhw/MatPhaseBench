import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

from MatPhaseBench_task_prompt_bank import (
    DIMENSION_DEFINITIONS,
    specified_dimension_description_prompt,
)
from vlm_json_parsing import parse_vlm_json_output_relaxed


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_SHIYUN_BASE_URL = "https://shiyunapi.com/v1"
DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


from llm_service import (
    call_vlm as llm_service_call_vlm,
    extract_vlm_output_text,
    extract_vlm_response_debug,
    is_empty_content_with_reasoning_json,
    is_finish_reason_length,
    parse_vlm_json_output as llm_service_parse_vlm_json_output,
    preview_json,
    preview_text,
)

INPUT_PATH = None
OUTPUT_PATH = None
PARTIAL_OUTPUT_PATH = None
FAILURE_OUTPUT_DIR = None
IMG_BASE_DIR = None
BACKEND = None
QWEN_LOCAL_MODEL = None
QWEN_LOCAL_BASE_URL = None
QWEN_LOCAL_API_KEY = None
DASHSCOPE_BASE_URL = None
DASHSCOPE_VL_MODEL = None
DASHSCOPE_API_KEY = None
SHIYUN_BASE_URL = None
SHIYUN_API_KEY = None
ZHIPU_BASE_URL = None
ZHIPU_API_KEY = None
ENABLE_THINKING = None
TEMPERATURE = None
TOP_P = None
MAX_TOKENS = None
TIMEOUT_SECONDS = None
RETRY_COUNT = None
RETRY_SLEEP_SECONDS = None
SAVE_INTERVAL = None
MAX_WORKERS = None
BASIC_CONFIG = None
REPROCESS_ABNORMAL = None


def parse_vlm_json_output(raw_text):
    return parse_vlm_json_output_relaxed(
        raw_text,
        strict_parser=llm_service_parse_vlm_json_output,
    )


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
        default=r"C:\Users\Wanghw\Desktop",
        help="Path to the image folder.",
    )
    parser.add_argument(
        "--input",
        default=str(ROOT_DIR / "dataset" / "image_match_evaluation_data_complete_info_200_classified.json"),
        help="Path to the classified input JSON file.",
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
        choices=("local", "dashscope", "shiyun", "zhipu"),
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
        "--shiyun-base-url",
        default=DEFAULT_SHIYUN_BASE_URL,
        help="Shiyun OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--shiyun-api-key",
        default=os.getenv("SHIYUN_API_KEY", ""),
        help="Shiyun API key. Defaults to the SHIYUN_API_KEY environment variable.",
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


def validate_backend_config(args):
    if args.backend != "local" and args.local_model:
        raise ValueError(
            f"--backend {args.backend} cannot be used together with --local-model. "
            "Remove --local-model or set --backend local."
        )
    if args.backend == "local" and not args.local_model:
        raise ValueError("--backend local requires a non-empty --local-model.")

def apply_runtime_config(args):
    global IMG_BASE_DIR
    global INPUT_PATH
    global OUTPUT_PATH
    global PARTIAL_OUTPUT_PATH
    global DASHSCOPE_API_KEY
    global FAILURE_OUTPUT_DIR
    global BACKEND
    global QWEN_LOCAL_MODEL
    global QWEN_LOCAL_BASE_URL
    global QWEN_LOCAL_API_KEY
    global DASHSCOPE_BASE_URL
    global DASHSCOPE_VL_MODEL
    global SHIYUN_BASE_URL
    global SHIYUN_API_KEY
    global ZHIPU_BASE_URL
    global ZHIPU_API_KEY
    global ENABLE_THINKING
    global TEMPERATURE
    global TOP_P
    global MAX_TOKENS
    global TIMEOUT_SECONDS
    global RETRY_COUNT
    global RETRY_SLEEP_SECONDS
    global SAVE_INTERVAL
    global MAX_WORKERS
    global BASIC_CONFIG
    global REPROCESS_ABNORMAL

    validate_backend_config(args)

    IMG_BASE_DIR = Path(args.img_base_dir)
    INPUT_PATH = Path(args.input)
    FAILURE_OUTPUT_DIR = Path(args.failure_output_dir)
    BACKEND = args.backend
    QWEN_LOCAL_MODEL = args.local_model
    QWEN_LOCAL_BASE_URL = args.local_base_url
    QWEN_LOCAL_API_KEY = args.local_api_key or "EMPTY"
    DASHSCOPE_BASE_URL = args.dashscope_base_url
    DASHSCOPE_VL_MODEL = args.model
    DASHSCOPE_API_KEY = args.dashscope_api_key
    SHIYUN_BASE_URL = args.shiyun_base_url
    SHIYUN_API_KEY = args.shiyun_api_key
    ZHIPU_BASE_URL = args.zhipu_base_url
    ZHIPU_API_KEY = args.zhipu_api_key
    ENABLE_THINKING = args.enable_thinking
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    TIMEOUT_SECONDS = args.timeout_seconds
    RETRY_COUNT = args.retry_count
    RETRY_SLEEP_SECONDS = args.retry_sleep_seconds
    SAVE_INTERVAL = args.save_interval
    MAX_WORKERS = args.max_workers
    REPROCESS_ABNORMAL = args.reprocess_abnormal
    BASIC_CONFIG = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "backend": BACKEND,
        "model": args.model,
        "local_model": args.local_model,
        "max_tokens": args.max_tokens,
        "max_workers": args.max_workers,
        "reprocess_abnormal": REPROCESS_ABNORMAL,
    }

    if args.output:
        OUTPUT_PATH = Path(args.output)
    else:
        output_model_name = QWEN_LOCAL_MODEL if BACKEND == "local" else DASHSCOPE_VL_MODEL
        OUTPUT_PATH = (
            Path(args.output_dir)
            / f"dimensions_description_model_{output_model_name}_think_{ENABLE_THINKING}.json"
        )

    PARTIAL_OUTPUT_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}.partial.json")


def print_runtime_config():
    print(
        "Runtime config: "
        f"input={INPUT_PATH} | "
        f"output={OUTPUT_PATH} | "
        f"backend={BACKEND} | "
        f"model={DASHSCOPE_VL_MODEL} | "
        f"local_model={QWEN_LOCAL_MODEL} | "
        f"max_tokens={MAX_TOKENS} | "
        f"max_workers={MAX_WORKERS} | "
        f"reprocess_abnormal={REPROCESS_ABNORMAL}",
        flush=True,
    )


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("JSON top level must be a list.")

    return data


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_selected_dimensions(selected_dimensions):
    if not selected_dimensions:
        raise ValueError("selected_dimensions cannot be empty.")

    invalid_aspects = set(selected_dimensions) - set(DIMENSION_DEFINITIONS)
    if invalid_aspects:
        raise ValueError(f"Invalid aspect labels: {sorted(invalid_aspects)}")


def build_selected_dimensions_text(selected_dimensions):
    return "\n".join(f"- {aspect}" for aspect in selected_dimensions)


def build_aspect_definitions_text(selected_dimensions):
    definition_blocks = []

    for aspect in selected_dimensions:
        definition_blocks.append(f"- {aspect}:\n  {DIMENSION_DEFINITIONS[aspect]}")

    return "\n\n".join(definition_blocks)


def build_output_schema(selected_dimensions):
    output_schema = {
        "selected_dimensions": selected_dimensions,
        "descriptions": {
            aspect: {
                "description": ""
            }
            for aspect in selected_dimensions
        },
        "comprehensive_description": "",
    }
    return json.dumps(output_schema, ensure_ascii=False, indent=2)


def build_specified_dimension_description_prompt(selected_dimensions):
    validate_selected_dimensions(selected_dimensions)

    return (
        specified_dimension_description_prompt
        .replace("{{SELECTED_DIMENSIONS}}", build_selected_dimensions_text(selected_dimensions))
        .replace("{{DIMENSION_DEFINITIONS}}", build_aspect_definitions_text(selected_dimensions))
        .replace("{{OUTPUT_SCHEMA}}", build_output_schema(selected_dimensions))
    )


def get_selected_dimensions(item):
    classification = item.get("dimension_multi_classification")
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


def build_image_path_from_base(image_path):
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image path must be a non-empty string.")

    image_path = image_path.strip()
    parsed = urlparse(image_path)
    if parsed.scheme in ("http", "https", "file"):
        return image_path

    path = Path(image_path)
    if path.is_absolute():
        try:
            path.relative_to(IMG_BASE_DIR)
            return str(path)
        except ValueError:
            return str(IMG_BASE_DIR / image_path.lstrip("/\\"))

    return str(IMG_BASE_DIR / image_path.lstrip("/\\"))


def get_record_image_path(item):
    image_path = item.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return image_path

    raise ValueError("Each input record must contain a non-empty image_path field.")


def call_vlm(prompt, image_path, backend=None):
    if backend is None:
        backend = BACKEND

    if backend == "local":
        model = QWEN_LOCAL_MODEL
        base_url = QWEN_LOCAL_BASE_URL
        api_key = QWEN_LOCAL_API_KEY or "EMPTY"
    elif backend == "dashscope":
        model = DASHSCOPE_VL_MODEL
        base_url = DASHSCOPE_BASE_URL
        api_key = DASHSCOPE_API_KEY
    elif backend == "shiyun":
        model = DASHSCOPE_VL_MODEL
        base_url = SHIYUN_BASE_URL
        api_key = SHIYUN_API_KEY
    elif backend == "zhipu":
        model = DASHSCOPE_VL_MODEL
        base_url = ZHIPU_BASE_URL
        api_key = ZHIPU_API_KEY
    else:
        raise ValueError("Unsupported backend. Use 'local', 'dashscope', 'shiyun', or 'zhipu'.")

    response = llm_service_call_vlm(
        prompt=prompt,
        backend=backend,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        enable_thinking=ENABLE_THINKING,
        timeout=TIMEOUT_SECONDS,
        top_k=20,
        image=image_path,
    )

    raw_text = response.get("raw_text", "") if isinstance(response, dict) else ""
    response_data = response.get("raw_response") if isinstance(response, dict) else None
    if not raw_text and response_data:
        raw_text = extract_vlm_output_text(response_data)

    if isinstance(response, dict) and response.get("success") is not True:
        if raw_text or response_data:
            return raw_text.strip(), response_data
        raise RuntimeError(
            "VLM service call failed: "
            f"backend={response.get('backend')} | "
            f"model={response.get('model')} | "
            f"error={response.get('error')}"
        )

    return raw_text.strip(), response_data


def make_safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def save_failure(sample_id, attempt, raw_text, raw_response, error=None):
    FAILURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FAILURE_OUTPUT_DIR / (
        f"sample_{make_safe_filename_part(sample_id)}_attempt_{attempt}.json"
    )
    response_debug = extract_vlm_response_debug(raw_response)
    write_json(
        {
            "sample_id": sample_id,
            "attempt": attempt,
            "basic_config": BASIC_CONFIG,
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
    )
    return output_path


def build_abnormal_phase_description_result(
    sample_id,
    image_path,
    selected_dimensions,
    failure_path,
    abnormal_reason,
):
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "selected_dimensions": selected_dimensions,
        "vlm_description": "Abnormal",
        "abnormal_reason": abnormal_reason,
        "failure_path": str(failure_path.resolve()),
    }


def get_processable_items(data):
    return [item for item in data if isinstance(item, dict)]


def build_sample_id_index(items):
    sample_id_to_item = {}
    duplicate_sample_ids = []
    for item in items:
        sample_id = item.get("sample_id")
        if sample_id in sample_id_to_item:
            duplicate_sample_ids.append(sample_id)
        sample_id_to_item[sample_id] = item

    if duplicate_sample_ids:
        raise ValueError(
            f"Duplicate sample_id values in INPUT: {sorted(duplicate_sample_ids)}"
        )

    return sample_id_to_item


def build_sample_id_to_position(items):
    sample_id_to_position = {}
    duplicate_sample_ids = []
    for index, item in enumerate(items):
        sample_id = item.get("sample_id")
        if sample_id in sample_id_to_position:
            duplicate_sample_ids.append(sample_id)
        sample_id_to_position[sample_id] = index

    if duplicate_sample_ids:
        raise ValueError(
            f"Duplicate sample_id values in INPUT: {sorted(duplicate_sample_ids)}"
        )

    return sample_id_to_position


def is_abnormal_result(record):
    return isinstance(record, dict) and record.get("vlm_description") == "Abnormal"


def build_ordered_partial_records(results, source_items=None):
    if not isinstance(results, dict):
        return results

    if source_items is None:
        return [results[index] for index in sorted(results)]

    return [
        results[index]
        for index in range(len(source_items))
        if index in results
    ]


def save_partial_results(results, source_items=None):
    partial_records = build_ordered_partial_records(results, source_items)
    write_json(partial_records, PARTIAL_OUTPUT_PATH)
    print(
        f"\nSaved partial VLM phase description result "
        f"({len(partial_records)} records) to {PARTIAL_OUTPUT_PATH.resolve()}"
    )


def load_resume_results(source_items):
    if not PARTIAL_OUTPUT_PATH.exists():
        return {}

    resume_results = load_json(PARTIAL_OUTPUT_PATH)
    if len(resume_results) > len(source_items):
        raise ValueError(
            "Partial output has more records than input data: "
            f"{PARTIAL_OUTPUT_PATH.resolve()}"
        )

    sample_id_to_index = build_sample_id_to_position(source_items)
    resume_results_by_index = {}
    duplicate_sample_ids = []

    for result_position, result in enumerate(resume_results):
        result_sample_id = result.get("sample_id") if isinstance(result, dict) else None
        if result_sample_id not in sample_id_to_index:
            raise ValueError(
                "Partial output contains a sample_id missing from input data at "
                f"partial_index={result_position}: partial sample_id={result_sample_id}, "
                f"file={PARTIAL_OUTPUT_PATH.resolve()}"
            )
        source_index = sample_id_to_index[result_sample_id]
        if source_index in resume_results_by_index:
            duplicate_sample_ids.append(result_sample_id)
        resume_results_by_index[source_index] = result

    if duplicate_sample_ids:
        raise ValueError(
            "Partial output contains duplicate sample_id values: "
            f"{sorted(duplicate_sample_ids)}"
        )

    print(f"Resuming from partial output: {PARTIAL_OUTPUT_PATH.resolve()}")
    print(f"Skipping {len(resume_results_by_index)} already processed records.")
    return resume_results_by_index


def remove_partial_output():
    if PARTIAL_OUTPUT_PATH.exists():
        PARTIAL_OUTPUT_PATH.unlink()
        print(f"Removed partial output: {PARTIAL_OUTPUT_PATH.resolve()}")


def describe_phase_diagram(item):
    sample_id = item.get("sample_id")
    record_image_path = get_record_image_path(item)
    image_path = build_image_path_from_base(record_image_path)
    selected_dimensions = get_selected_dimensions(item)
    prompt = build_specified_dimension_description_prompt(selected_dimensions)

    last_error = None
    last_raw_text = ""
    last_raw_response = None
    failure_paths = []
    for attempt in range(1, RETRY_COUNT + 1):
        raw_text = ""
        raw_response = None
        try:
            # print("[Debug] prompt = ", prompt)
            raw_text, raw_response = call_vlm(prompt, image_path, BACKEND)
            last_raw_text = raw_text
            last_raw_response = raw_response
            parsed_result = parse_vlm_json_output(raw_text)
            # print("[Debug] parsed_result = ", parsed_result)

            return {
                "sample_id": sample_id,
                "image_path": record_image_path,
                "selected_dimensions": selected_dimensions,
                "vlm_description": parsed_result,
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
                error=error,
            )
            failure_paths.append(failure_path)

            if is_empty_content_with_reasoning_json(raw_text, raw_response):
                return build_abnormal_phase_description_result(
                    sample_id=sample_id,
                    image_path=record_image_path,
                    selected_dimensions=selected_dimensions,
                    failure_path=failure_path,
                    abnormal_reason=(
                        "message.content is empty while reasoning_content contains JSON. "
                        "Only message.content is accepted as the final answer."
                    ),
                )

            if is_finish_reason_length(raw_response):
                return build_abnormal_phase_description_result(
                    sample_id=sample_id,
                    image_path=record_image_path,
                    selected_dimensions=selected_dimensions,
                    failure_path=failure_path,
                    abnormal_reason=(
                        "finish_reason is length, indicating the model output was "
                        "truncated before a valid JSON answer was available."
                    ),
                )

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_SLEEP_SECONDS)

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


def process_data(data):
    source_items = get_processable_items(data)
    completed_results = load_resume_results(source_items)
    pending_items = [
        (index, item)
        for index, item in enumerate(source_items)
        if index not in completed_results
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(describe_phase_diagram, item): index
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
                    save_partial_results(completed_results, source_items)
                    raise

                progress_bar.update(1)

                if len(completed_results) % SAVE_INTERVAL == 0:
                    save_partial_results(completed_results, source_items)

    if completed_results and len(completed_results) % SAVE_INTERVAL != 0:
        save_partial_results(completed_results, source_items)

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


def load_error_reprocessing_base_results():
    if PARTIAL_OUTPUT_PATH.exists():
        partial_results = load_json(PARTIAL_OUTPUT_PATH)
        print(f"Resuming error reprocessing from partial output: {PARTIAL_OUTPUT_PATH.resolve()}")
        return partial_results

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            "Cannot reprocess Abnormal records because previous output does not exist: "
            f"{OUTPUT_PATH.resolve()}"
        )

    initial_results = load_json(OUTPUT_PATH)
    print(
        "Loaded previous output for Abnormal reprocessing: "
        f"{OUTPUT_PATH.resolve()}"
    )
    return initial_results


def process_error_reprocessing(data):
    source_items = get_processable_items(data)
    sample_id_to_item = build_sample_id_index(source_items)
    results = load_error_reprocessing_base_results()

    targets = []
    missing_sample_ids = []
    for index, result in enumerate(results):
        if not is_abnormal_result(result):
            continue

        sample_id = result.get("sample_id")
        source_item = sample_id_to_item.get(sample_id)
        if source_item is None:
            missing_sample_ids.append(sample_id)
            continue
        targets.append((index, source_item))

    if missing_sample_ids:
        raise ValueError(
            "Previous output contains Abnormal sample_id values "
            f"missing from INPUT: {missing_sample_ids}"
        )

    print(f"Found {len(targets)} Abnormal records to reprocess.")
    if not targets:
        return results

    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(describe_phase_diagram, source_item): index
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
                    save_partial_results(results)
                    raise

                processed_count += 1
                progress_bar.update(1)
                if processed_count % SAVE_INTERVAL == 0:
                    save_partial_results(results)

    if processed_count % SAVE_INTERVAL != 0:
        save_partial_results(results)

    return results


def main():
    args = parse_args()
    apply_runtime_config(args)
    print_runtime_config()

    data = load_json(INPUT_PATH)
    if REPROCESS_ABNORMAL:
        results = process_error_reprocessing(data)
    else:
        results = process_data(data)

    write_json(results, OUTPUT_PATH)
    print(f"Wrote VLM phase description result to {OUTPUT_PATH.resolve()}")
    remove_partial_output()


if __name__ == "__main__":
    main()
