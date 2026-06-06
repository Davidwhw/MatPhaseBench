from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SHIYUN_BASE_URL = "https://shiyunapi.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

QWEN_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_LOCAL_API_KEY = "EMPTY"

VALID_BACKENDS = {"local", "dashscope", "shiyun", "zhipu"}
SHIYUN_GPT_REASONING_PAYLOAD = {
    "effort": "medium",
    "summary": "auto",
}
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


def is_invalid_escape_error(error: json.JSONDecodeError) -> bool:
    return "Invalid \\escape" in str(error) or "Invalid \\escape" in error.msg


def skip_whitespace(text: str, start_index: int) -> int:
    index = start_index
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def parse_json_string_sequence(text: str) -> list[str] | None:
    values: list[str] = []
    index = skip_whitespace(text, 0)

    while index < len(text):
        if text[index] != '"':
            return None

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
            return None

        try:
            value = json.loads(text[literal_start:index])
        except json.JSONDecodeError:
            return None

        if not isinstance(value, str):
            return None
        values.append(value)

        index = skip_whitespace(text, index)
        if index >= len(text):
            return values
        if text[index] != ",":
            return None
        index = skip_whitespace(text, index + 1)

    return values


def repair_extra_string_values_after_key(text: str) -> str:
    pattern = re.compile(r'("(?:(?:\\.)|[^"\\])+"\s*:\s*)')
    output: list[str] = []
    cursor = 0

    for match in pattern.finditer(text):
        value_start = match.end()
        values, value_end = parse_json_string_sequence_at(text, value_start)
        if values is None or len(values) <= 1:
            continue

        next_index = skip_whitespace(text, value_end)
        if next_index < len(text) and text[next_index] not in ",]}":
            continue

        output.append(text[cursor:value_start])
        output.append(json.dumps("; ".join(values), ensure_ascii=False))
        cursor = value_end

    if not output:
        return text

    output.append(text[cursor:])
    return "".join(output)


def parse_json_string_sequence_at(text: str, start_index: int) -> tuple[list[str] | None, int]:
    values: list[str] = []
    index = skip_whitespace(text, start_index)

    while index < len(text):
        if text[index] != '"':
            return (values, index) if values else (None, start_index)

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
            return None, start_index

        try:
            value = json.loads(text[literal_start:index])
        except json.JSONDecodeError:
            return None, start_index

        if not isinstance(value, str):
            return None, start_index
        values.append(value)

        after_value = skip_whitespace(text, index)
        if after_value >= len(text):
            return values, after_value
        if text[after_value] != ",":
            return values, after_value

        next_index = skip_whitespace(text, after_value + 1)
        if next_index >= len(text) or text[next_index] != '"':
            return values, after_value

        next_literal_end = find_json_string_end(text, next_index)
        if next_literal_end is None:
            return None, start_index

        after_next_literal = skip_whitespace(text, next_literal_end)
        if after_next_literal < len(text) and text[after_next_literal] == ":":
            return values, after_value

        index = next_index

    return values, index


def find_json_string_end(text: str, start_index: int) -> int | None:
    if start_index >= len(text) or text[start_index] != '"':
        return None

    index = start_index + 1
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index + 1
        index += 1

    return None


def loads_json_with_repairs(text: str) -> Any:
    candidates = [
        text,
        escape_invalid_backslashes(text),
        repair_extra_string_values_after_key(text),
        escape_invalid_backslashes(repair_extra_string_values_after_key(text)),
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
        if is_invalid_escape_error(first_error):
            return loads_json_with_repairs(escape_invalid_backslashes(raw))

    json_match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object or array found in LLM output.")

    json_text = json_match.group(1)
    try:
        return loads_json_with_repairs(json_text)
    except json.JSONDecodeError as second_error:
        if is_invalid_escape_error(second_error):
            return loads_json_with_repairs(escape_invalid_backslashes(json_text))
        raise ValueError(f"Failed to parse JSON after extraction: {second_error}") from second_error


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


def stringify_message_field(value: Any) -> str:
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


def _extract_reasoning_output_text(response_data: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for output_item in response_data.get("output", []):
        if not isinstance(output_item, dict):
            continue
        if output_item.get("type") != "reasoning":
            continue
        for summary_item in output_item.get("summary", []):
            if not isinstance(summary_item, dict):
                continue
            text = summary_item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        encrypted_content = output_item.get("encrypted_content")
        if encrypted_content and not text_parts:
            text_parts.append(str(encrypted_content))
    return "\n".join(text_parts)


def extract_vlm_response_debug(response_data: Any) -> dict[str, Any]:
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
            "content": stringify_message_field(message.get("content")).strip(),
            "reasoning_content": stringify_message_field(
                message.get("reasoning_content")
            ).strip(),
            "reasoning": stringify_message_field(message.get("reasoning")).strip(),
        }

    reasoning = response_data.get("reasoning")
    if isinstance(reasoning, (dict, list)):
        reasoning_text = json.dumps(reasoning, ensure_ascii=False)
    else:
        reasoning_text = stringify_message_field(reasoning).strip()

    return {
        "finish_reason": response_data.get("status"),
        "content": _extract_responses_text(response_data).strip(),
        "reasoning_content": _extract_reasoning_output_text(response_data).strip(),
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


def reasoning_content_contains_json(raw_response: Any) -> bool:
    response_debug = extract_vlm_response_debug(raw_response)
    reasoning_content = response_debug.get("reasoning_content") or ""
    return bool(extract_first_json_value(reasoning_content))


def is_empty_content_with_reasoning_json(raw_text: Any, raw_response: Any) -> bool:
    if raw_text and str(raw_text).strip():
        return False

    response_debug = extract_vlm_response_debug(raw_response)
    content = response_debug.get("content") or ""
    if content.strip():
        return False

    return reasoning_content_contains_json(raw_response)


def is_finish_reason_length(raw_response: Any) -> bool:
    response_debug = extract_vlm_response_debug(raw_response)
    finish_reason = response_debug.get("finish_reason")
    return str(finish_reason).lower() == "length"


def _guess_image_mime_type(local_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(local_path))
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "image/jpeg"


def _build_data_url_from_local_file(local_path: Path) -> str:
    if not local_path.is_file():
        raise FileNotFoundError(f"Local image not found: {local_path}")

    file_size = local_path.stat().st_size
    if file_size > 7 * 1024 * 1024:
        raise ValueError(
            f"Local image is too large ({file_size} bytes). "
            "It may exceed the backend limit after base64 encoding."
        )

    mime_type = _guess_image_mime_type(local_path)
    encoded = base64.b64encode(local_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _local_file_from_source(image_source: str) -> Path:
    parsed = urlparse(image_source)
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    return Path(image_source).expanduser().resolve()


def _normalize_local_image_source(image_source: str) -> str:
    if not isinstance(image_source, str) or not image_source.strip():
        raise ValueError("image must be a non-empty string.")

    image_source = image_source.strip()
    parsed = urlparse(image_source)
    if parsed.scheme in ("http", "https", "data"):
        return image_source

    local_path = _local_file_from_source(image_source)
    return _build_data_url_from_local_file(local_path)


def _normalize_dashscope_image_source(image_source: str) -> str:
    if not isinstance(image_source, str) or not image_source.strip():
        raise ValueError("image must be a non-empty string.")

    image_source = image_source.strip()
    parsed = urlparse(image_source)
    if parsed.scheme in ("http", "https"):
        return image_source

    local_path = _local_file_from_source(image_source)
    return _build_data_url_from_local_file(local_path)


def _build_messages(
    prompt: str,
    *,
    backend: str,
    system_prompt: str | None = None,
    image: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if not image:
        messages.append({"role": "user", "content": prompt})
        return messages

    if backend == "local":
        image_url = _normalize_local_image_source(image)
    elif backend in {"dashscope", "shiyun", "zhipu"}:
        image_url = _normalize_dashscope_image_source(image)
    else:
        raise ValueError(
            "Unsupported backend. Use 'local', 'dashscope', 'shiyun', or 'zhipu'."
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    )
    return messages


def _extract_reasoning(message: dict[str, Any], enable_thinking: bool) -> Any:
    if not enable_thinking:
        return None
    return message.get("reasoning") or message.get("reasoning_content")


def _is_shiyun_gpt_model(model: str | None) -> bool:
    return isinstance(model, str) and model.strip().lower().startswith("gpt")


def _build_responses_input(
    prompt: str,
    *,
    image: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]
    if image:
        content.append(
            {
                "type": "input_image",
                "image_url": _normalize_dashscope_image_source(image),
            }
        )
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


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


def _error_response(
    *,
    backend: str,
    model: str | None,
    prompt: str,
    enable_thinking: bool,
    image: str | None,
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


def call_vlm(
    prompt: str,
    backend: str = "local",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 20480,
    enable_thinking: bool = False,
    timeout: int | None = 1000,
    presence_penalty: float = 1.5,
    top_k: int = 20,
    system_prompt: str | None = None,
    image: str | None = None,
) -> dict[str, Any]:
    """
    Call a local, DashScope, Shiyun, or Zhipu chat completion API.

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
            error="Unsupported backend. Use 'local', 'dashscope', 'shiyun', or 'zhipu'.",
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
        use_base_url = base_url or QWEN_LOCAL_BASE_URL
        use_api_key = api_key or QWEN_LOCAL_API_KEY
        api_key_env_name = None
        extra_payload: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            },
            "top_k": top_k,
            "presence_penalty": presence_penalty,
        }
    elif backend == "dashscope":
        use_base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL
        use_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        api_key_env_name = "DASHSCOPE_API_KEY"
        extra_payload = {"enable_thinking": enable_thinking}
    elif backend == "shiyun":
        use_base_url = base_url or os.getenv("SHIYUN_BASE_URL") or SHIYUN_BASE_URL
        use_api_key = api_key or os.getenv("SHIYUN_API_KEY")
        api_key_env_name = "SHIYUN_API_KEY"
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
        use_responses_api = (
            backend == "shiyun"
            and enable_thinking
            and _is_shiyun_gpt_model(model)
        )
        if use_responses_api:
            payload = {
                "model": model,
                "input": _build_responses_input(prompt, image=image),
                "text": {
                    "format": {
                        "type": "text",
                    },
                    "verbosity": "medium",
                },
                "reasoning": dict(SHIYUN_GPT_REASONING_PAYLOAD),
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
                "store": True,
            }
            if system_prompt:
                payload["instructions"] = system_prompt
            endpoint_path = "responses"
        else:
            messages = _build_messages(
                prompt,
                backend=backend,
                system_prompt=system_prompt,
                image=image,
            )
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                **extra_payload,
            }
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

        if use_responses_api:
            raw_text = _extract_responses_text(response_data)
            reasoning = response_data.get("reasoning")
            finish_reason = response_data.get("status")
        else:
            choice = response_data["choices"][0]
            message = choice.get("message", {}) or {}
            raw_text = message.get("content", "") or ""
            reasoning = _extract_reasoning(message, enable_thinking)
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


llm_infer = call_vlm


def require_llm_json_answer(response: dict[str, Any], stage_name: str = "LLM call") -> dict[str, Any]:
    if not isinstance(response, dict):
        raise RuntimeError(f"{stage_name} failed: response is not a dict: {type(response)}")

    if response.get("success") is not True:
        raw_text = response.get("raw_text", "")
        if isinstance(raw_text, str) and len(raw_text) > 1000:
            raw_text = raw_text[:1000] + "...<truncated>"
        raise RuntimeError(
            f"{stage_name} failed: backend={response.get('backend')} | "
            f"model={response.get('model')} | error={response.get('error')} | "
            f"raw_text={raw_text}"
        )

    answer = response.get("answer")
    if not isinstance(answer, dict):
        raise RuntimeError(
            f"{stage_name} failed: parsed answer is not a dict: {type(answer)}"
        )
    return answer
