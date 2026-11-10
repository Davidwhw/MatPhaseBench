from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from src.utils import prepare_image
except ModuleNotFoundError:
    from utils import prepare_image


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
JIEKOU_BASE_URL = "https://api.highwayapi.ai/openai"
ICOMPIFY_BASE_URL = "https://api.icompify.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
TOKENRIVER_BASE_URL = "https://api.tokenriver.cn/v1"
TENCENT_BASE_URL = "https://tokenhub.tencentmaas.com/plan/v3"
CNIC_BASE_URL = "http://api.aicnic.cn/v1"

QWEN_LOCAL_BASE_URL = os.getenv("QWEN_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_LOCAL_API_KEY = "EMPTY"

VALID_BACKENDS = {
    "local",
    "dashscope",
    "jiekou",
    "icompify",
    "zhipu",
    "tokenriver",
    "tencent",
    "cnic",
}
QWEN_LOCAL_REASONING_EFFORTS = {"xhigh", "medium", "low"}
VALID_JSON_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t"}


def escape_invalid_backslashes(text: str) -> str:
    output: list[str] = []
    index = 0

    while index < len(text):
        char = text[index]
        if char != "\\":
            output.append(char)
            index += 1
            continue

        if index + 1 >= len(text):
            output.append("\\\\")
            index += 1
            continue

        next_char = text[index + 1]
        if next_char in VALID_JSON_ESCAPES:
            output.append("\\")
            output.append(next_char)
            index += 2
            continue

        if next_char == "u" and index + 5 < len(text):
            hex_part = text[index + 2 : index + 6]
            if all(char in "0123456789abcdefABCDEF" for char in hex_part):
                output.append("\\")
                output.append("u")
                output.append(hex_part)
                index += 6
                continue

        output.append("\\\\")
        index += 1

    return "".join(output)


def repair_extra_string_values_after_key(text: str) -> str:
    pattern = re.compile(r'("(?:(?:\\.)|[^"\\])+"\s*:\s*)')
    output: list[str] = []
    cursor = 0

    for match in pattern.finditer(text):
        value_start = match.end()
        values: list[str] = []
        index = value_start
        while index < len(text) and text[index].isspace():
            index += 1

        while index < len(text):
            if text[index] != '"':
                values, value_end = (values, index) if values else (None, value_start)
                break

            literal_start = index
            index += 1
            escaped = False
            while index < len(text):
                char = text[index]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    index += 1
                    break
                index += 1
            else:
                values, value_end = None, value_start
                break

            try:
                value = json.loads(text[literal_start:index])
            except json.JSONDecodeError:
                values, value_end = None, value_start
                break

            if not isinstance(value, str):
                values, value_end = None, value_start
                break
            values.append(value)

            after_value = index
            while after_value < len(text) and text[after_value].isspace():
                after_value += 1
            if after_value >= len(text) or text[after_value] != ",":
                value_end = after_value
                break

            next_index = after_value + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index >= len(text) or text[next_index] != '"':
                value_end = after_value
                break

            next_literal_end = next_index + 1
            escaped = False
            while next_literal_end < len(text):
                char = text[next_literal_end]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    next_literal_end += 1
                    break
                next_literal_end += 1
            else:
                values, value_end = None, value_start
                break

            after_next_literal = next_literal_end
            while (
                after_next_literal < len(text)
                and text[after_next_literal].isspace()
            ):
                after_next_literal += 1
            if after_next_literal < len(text) and text[after_next_literal] == ":":
                value_end = after_value
                break

            index = next_index
        else:
            value_end = index

        if values is None or len(values) <= 1:
            continue

        next_index = value_end
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        if next_index < len(text) and text[next_index] not in ",]}":
            continue

        output.append(text[cursor:value_start])
        output.append(json.dumps("; ".join(values), ensure_ascii=False))
        cursor = value_end

    if not output:
        return text

    output.append(text[cursor:])
    return "".join(output)


def loads_json_with_repairs(text: str) -> Any:
    repaired_text = repair_extra_string_values_after_key(text)
    candidates = [
        text,
        escape_invalid_backslashes(text),
        repaired_text,
        escape_invalid_backslashes(repaired_text),
    ]

    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = error

    if last_error is None:
        raise ValueError("No JSON parse candidate was generated.")
    raise last_error


def parse_llm_json_output(text: str) -> Any:
    """
    Parse JSON returned by an LLM.

    Handles raw JSON, fenced ```json blocks, text wrapped around a JSON object or
    array, invalid backslashes, and simple repeated string values after one key.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    raw = text.strip()
    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        raw = fence_match.group(1).strip()

    try:
        return loads_json_with_repairs(raw)
    except json.JSONDecodeError as first_error:
        if (
            "Invalid \\escape" in str(first_error)
            or "Invalid \\escape" in first_error.msg
        ):
            return loads_json_with_repairs(escape_invalid_backslashes(raw))

    json_match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object or array found in LLM output.")

    json_text = json_match.group(1)
    try:
        return loads_json_with_repairs(json_text)
    except json.JSONDecodeError as second_error:
        if (
            "Invalid \\escape" in str(second_error)
            or "Invalid \\escape" in second_error.msg
        ):
            return loads_json_with_repairs(escape_invalid_backslashes(json_text))
        raise ValueError(
            f"Failed to parse JSON after extraction: {second_error}"
        ) from second_error


def extract_first_json_value(text: str) -> str | None:
    start_candidates = [
        index
        for index in (text.find("{"), text.find("["))
        if index != -1
    ]
    if not start_candidates:
        return None

    start = min(start_candidates)
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[start : index + 1]

    return None


def parse_vlm_json_output(text: str) -> Any:
    """
    Parse the final VLM answer as JSON.

    This intentionally parses only the final answer text. Reasoning fields are
    diagnostic metadata and should not be used as the JSON answer.
    """
    if not isinstance(text, str):
        raise TypeError("LLM output must be a string.")

    raw = text.strip()
    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        raw = fence_match.group(1).strip()

    json_text = extract_first_json_value(raw) or raw
    return loads_json_with_repairs(json_text)


def extract_vlm_response_debug(response_data: Any) -> dict[str, Any]:
    def stringify_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(value)

    if not isinstance(response_data, dict):
        return {
            "finish_reason": None,
            "content": "",
            "reasoning_content": "",
            "reasoning": "",
        }

    choices = response_data.get("choices") or []
    if choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message", {}) or {}
        return {
            "finish_reason": choice.get("finish_reason"),
            "content": stringify_field(message.get("content")).strip(),
            "reasoning_content": stringify_field(
                message.get("reasoning_content")
            ).strip(),
            "reasoning": stringify_field(message.get("reasoning")).strip(),
        }

    reasoning = response_data.get("reasoning")
    if isinstance(reasoning, (dict, list)):
        reasoning_text = json.dumps(reasoning, ensure_ascii=False)
    else:
        reasoning_text = stringify_field(reasoning).strip()

    reasoning_parts: list[str] = []
    for output_item in response_data.get("output", []):
        if not isinstance(output_item, dict) or output_item.get("type") != "reasoning":
            continue
        for summary_item in output_item.get("summary", []):
            if not isinstance(summary_item, dict):
                continue
            text = summary_item.get("text")
            if isinstance(text, str) and text.strip():
                reasoning_parts.append(text.strip())
        encrypted_content = output_item.get("encrypted_content")
        if encrypted_content and not reasoning_parts:
            reasoning_parts.append(str(encrypted_content))

    return {
        "finish_reason": response_data.get("status"),
        "content": _extract_responses_text(response_data).strip(),
        "reasoning_content": "\n".join(reasoning_parts).strip(),
        "reasoning": reasoning_text,
    }


def extract_vlm_output_text(response_data: Any) -> str:
    """
    Return only the final answer content.

    In think mode, reasoning fields are useful for diagnostics but must not be
    parsed as the final JSON answer.
    """
    debug = extract_vlm_response_debug(response_data)
    return (debug.get("content") or "").strip()


def preview_text(text: Any, limit: int = 2000) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def preview_json(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return preview_text(text, limit=limit)


def is_empty_content_with_reasoning_json(raw_text: Any, raw_response: Any) -> bool:
    if raw_text and str(raw_text).strip():
        return False

    response_debug = extract_vlm_response_debug(raw_response)
    content = response_debug.get("content") or ""
    if content.strip():
        return False

    reasoning_content = response_debug.get("reasoning_content") or ""
    return bool(extract_first_json_value(reasoning_content))


def is_finish_reason_length(raw_response: Any) -> bool:
    response_debug = extract_vlm_response_debug(raw_response)
    finish_reason = response_debug.get("finish_reason")
    return str(finish_reason).lower() == "length"


def _normalize_image_source(image_source: str, *, allow_data_url: bool = False) -> str:
    if not isinstance(image_source, str) or not image_source.strip():
        raise ValueError("image must be a non-empty string.")

    image_source = image_source.strip()
    parsed = urlparse(image_source)
    if parsed.scheme in ("http", "https") or (
        allow_data_url and parsed.scheme == "data"
    ):
        return image_source

    if parsed.scheme == "file":
        local_path = Path(parsed.path).expanduser().resolve()
    else:
        local_path = Path(image_source).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"Local image not found: {local_path}")

    file_size = local_path.stat().st_size
    if file_size > 7 * 1024 * 1024:
        raise ValueError(
            f"Local image is too large ({file_size} bytes). "
            "It may exceed the backend limit after base64 encoding."
        )

    return prepare_image(local_path)


def _normalize_image_inputs(image: str | Sequence[str] | None) -> list[str]:
    if image is None:
        return []

    if isinstance(image, str):
        if not image.strip():
            raise ValueError("image must be a non-empty string.")
        return [image]

    if isinstance(image, (bytes, bytearray)) or not isinstance(image, Sequence):
        raise ValueError(
            "image must be a string path/URL or a non-empty sequence of "
            "image paths/URLs."
        )

    if not image:
        raise ValueError("image sequence must not be empty.")

    image_sources = []
    for index, image_source in enumerate(image):
        if not isinstance(image_source, str) or not image_source.strip():
            raise ValueError(
                "image sequence entries must be non-empty strings; "
                f"invalid entry at index {index}."
            )
        image_sources.append(image_source)

    return image_sources


def _extract_responses_text(response_data: dict[str, Any]) -> str:
    output_text = response_data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    text_parts: list[str] = []
    for output_item in response_data.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    return "\n".join(text_parts)


def _is_glm_model(model: str) -> bool:
    return model.strip().lower().startswith("glm")


def _error_response(
    *,
    backend: str,
    model: str | None,
    prompt: str,
    enable_thinking: bool,
    image: str | Sequence[str] | None,
    error: str,
    raw_text: str = "",
    raw_response: Any = None,
    reasoning: Any = None,
    finish_reason: Any = None,
    usage: Any = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "backend": backend,
        "model": model,
        "prompt": prompt,
        "enable_thinking": enable_thinking,
        "is_multimodal": bool(image),
        "image": image,
        "error": error,
        "raw_text": raw_text,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": raw_response,
    }


def vlm_client(
    prompt: str,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int | None = None,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
    image: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Call a local, DashScope, Jiekou, Icompify, Zhipu, TokenRiver, Tencent, or CNIC chat completion API.

    Returns a uniform dict. On success, ``answer`` is the parsed JSON result from
    the model content. On failure, ``error`` and the available raw fields are
    returned for retry or diagnostics.
    """
    if backend not in VALID_BACKENDS:
        return _error_response(
            backend=backend,
            model=model,
            prompt=prompt,
            enable_thinking=enable_thinking,
            image=image,
            error=(
                "Unsupported backend. Use 'local', 'dashscope', 'jiekou', "
                "'icompify', 'zhipu', 'tokenriver', 'tencent', or 'cnic'."
            ),
        )

    if not isinstance(prompt, str) or not prompt.strip():
        return _error_response(
            backend=backend,
            model=model,
            prompt=prompt,
            enable_thinking=enable_thinking,
            image=image,
            error="prompt must be a non-empty string.",
        )

    if not isinstance(model, str) or not model.strip():
        return _error_response(
            backend=backend,
            model=model,
            prompt=prompt,
            enable_thinking=enable_thinking,
            image=image,
            error="model is required. Pass an explicit model name.",
        )
    model = model.strip()

    if backend == "local":
        use_reasoning_effort = reasoning_effort or "medium"
        if use_reasoning_effort not in QWEN_LOCAL_REASONING_EFFORTS:
            return _error_response(
                backend=backend,
                model=model,
                prompt=prompt,
                enable_thinking=enable_thinking,
                image=image,
                error=(
                    "local Qwen reasoning_effort must be one of "
                    f"{sorted(QWEN_LOCAL_REASONING_EFFORTS)}."
                ),
            )

        use_base_url = base_url or QWEN_LOCAL_BASE_URL
        use_api_key = api_key or QWEN_LOCAL_API_KEY
        api_key_env_name = None
        extra_payload: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
                "reasoning_effort": use_reasoning_effort,
            },
            "top_k": top_k,
            "presence_penalty": presence_penalty,
        }
    elif backend == "dashscope":
        use_base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL
        use_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        api_key_env_name = "DASHSCOPE_API_KEY"
        extra_payload = {"enable_thinking": enable_thinking}
    elif backend == "jiekou":
        use_base_url = base_url or os.getenv("JIEKOU_BASE_URL") or JIEKOU_BASE_URL
        use_api_key = api_key or os.getenv("JIEKOU_API_KEY")
        api_key_env_name = "JIEKOU_API_KEY"
        extra_payload = {}
    elif backend == "icompify":
        use_base_url = base_url or os.getenv("ICOMPIFY_BASE_URL") or ICOMPIFY_BASE_URL
        use_api_key = api_key or os.getenv("ICOMPIFY_API_KEY")
        api_key_env_name = "ICOMPIFY_API_KEY"
        extra_payload = {}
    elif backend == "tokenriver":
        use_base_url = (
            base_url
            or os.getenv("TOKENRIVER_BASE_URL")
            or TOKENRIVER_BASE_URL
        )
        use_api_key = api_key or os.getenv("TOKENRIVER_API_KEY")
        api_key_env_name = "TOKENRIVER_API_KEY"
        extra_payload = {}
    elif backend == "tencent":
        use_base_url = base_url or os.getenv("TENCENT_BASE_URL") or TENCENT_BASE_URL
        use_api_key = api_key or os.getenv("TENCENT_API_KEY")
        api_key_env_name = "TENCENT_API_KEY"
        extra_payload = {"stream": False}
        if _is_glm_model(model):
            extra_payload["reasoning_effort"] = reasoning_effort or "medium"
    elif backend == "cnic":
        use_base_url = base_url or os.getenv("CNIC_BASE_URL") or CNIC_BASE_URL
        use_api_key = api_key or os.getenv("CNIC_API_KEY")
        api_key_env_name = "CNIC_API_KEY"
        extra_payload = {}
    else:
        use_base_url = base_url or os.getenv("ZHIPU_BASE_URL") or ZHIPU_BASE_URL
        use_api_key = api_key or os.getenv("ZHIPU_API_KEY")
        api_key_env_name = "ZHIPU_API_KEY"
        extra_payload = {}
        if enable_thinking:
            extra_payload["thinking"] = {"type": "enabled"}

    if not use_api_key:
        if api_key_env_name:
            error = (
                f"{backend} API key is missing. Provide api_key or set "
                f"{api_key_env_name}."
            )
        else:
            error = f"{backend} API key is missing. Provide api_key."
        return _error_response(
            backend=backend,
            model=model,
            prompt=prompt,
            enable_thinking=enable_thinking,
            image=image,
            error=error,
        )

    try:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        image_sources = _normalize_image_inputs(image)
        if image_sources:
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _normalize_image_source(
                            image_source,
                            allow_data_url=(backend in {"local", "cnic"}),
                        )
                    },
                }
                for image_source in image_sources
            ]
            content.append({"type": "text", "text": prompt})
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            **extra_payload,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        endpoint_path = "chat/completions"

        request = urllib.request.Request(
            f"{use_base_url.rstrip('/')}/{endpoint_path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {use_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        choice = response_data["choices"][0]
        message = choice.get("message", {}) or {}
        raw_text = message.get("content", "") or ""
        reasoning = (
            message.get("reasoning") or message.get("reasoning_content")
            if enable_thinking
            else None
        )
        finish_reason = choice.get("finish_reason")
        usage = response_data.get("usage")

        try:
            parsed_answer = parse_llm_json_output(raw_text)
        except Exception as parse_error:
            return _error_response(
                backend=backend,
                model=response_data.get("model", model),
                prompt=prompt,
                enable_thinking=enable_thinking,
                image=image,
                error=f"JSON parse failed: {parse_error}",
                raw_text=raw_text,
                raw_response=response_data,
                reasoning=reasoning,
                finish_reason=finish_reason,
                usage=usage,
            )

        return {
            "success": True,
            "backend": backend,
            "model": response_data.get("model", model),
            "prompt": prompt,
            "enable_thinking": enable_thinking,
            "is_multimodal": bool(image),
            "image": image,
            "answer": parsed_answer,
            "raw_text": raw_text,
            "reasoning": reasoning,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw_response": response_data,
        }
    except urllib.error.HTTPError as error:
        raw_error_body = error.read().decode("utf-8", errors="replace")
        try:
            response_data = json.loads(raw_error_body) if raw_error_body else None
        except json.JSONDecodeError:
            response_data = raw_error_body
        return _error_response(
            backend=backend,
            model=model,
            prompt=prompt,
            enable_thinking=enable_thinking,
            image=image,
            error=(
                f"HTTP Error {error.code}: {error.reason}; "
                f"response_body={raw_error_body}"
            ),
            raw_text=raw_error_body,
            raw_response=response_data,
        )
    except Exception as error:
        return _error_response(
            backend=backend,
            model=model,
            prompt=prompt,
            enable_thinking=enable_thinking,
            image=image,
            error=str(error),
            raw_text=locals().get("raw_text", ""),
            raw_response=locals().get("response_data"),
        )
