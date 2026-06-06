import json
import re

JSON_ESCAPE_CHARS = set('"\\/bfnrtu')

# JSON treats \b, \f, \n, \r, and \t as valid escapes, so LaTeX commands
# beginning with those letters need explicit repair before json.loads().
LATEX_COMMANDS_REQUIRING_ESCAPE = {
    "backslash",
    "bar",
    "begin",
    "beta",
    "bf",
    "big",
    "Big",
    "bigg",
    "Bigg",
    "boldsymbol",
    "boxed",
    "bmatrix",
    "fbox",
    "forall",
    "frac",
    "ne",
    "neq",
    "nabla",
    "nu",
    "not",
    "rightarrow",
    "rangle",
    "ref",
    "rho",
    "right",
    "rm",
    "text",
    "theta",
    "tilde",
    "times",
    "tau",
}


def strip_markdown_json_fence(text):
    if not isinstance(text, str):
        return text

    stripped = text.strip()
    fence_match = re.fullmatch(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    return stripped


def _repair_json_string_backslashes(json_text):
    if not isinstance(json_text, str):
        return json_text

    repaired = []
    in_string = False
    i = 0

    while i < len(json_text):
        char = json_text[i]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            i += 1
            continue

        if char != "\\":
            repaired.append(char)
            i += 1
            continue

        if i == len(json_text) - 1:
            repaired.append("\\\\")
            i += 1
            continue

        next_char = json_text[i + 1]
        if next_char.isalpha():
            command_end = i + 1
            while (
                command_end < len(json_text)
                and json_text[command_end].isalpha()
            ):
                command_end += 1
            command = json_text[i + 1:command_end]
            if command in LATEX_COMMANDS_REQUIRING_ESCAPE:
                repaired.append("\\\\")
                i += 1
                continue

        if next_char not in JSON_ESCAPE_CHARS:
            repaired.append("\\\\")
            i += 1
            continue

        repaired.append(char)
        repaired.append(next_char)
        i += 2

    return "".join(repaired)


def _has_semdesc_output_shape(parsed):
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("selected_dimensions"), list)
        and isinstance(parsed.get("descriptions"), dict)
        and isinstance(parsed.get("comprehensive_description"), str)
    )


def _load_semdesc_json_candidate(json_text):
    try:
        parsed = json.loads(json_text, strict=False)
    except Exception:
        return None

    if _has_semdesc_output_shape(parsed):
        return parsed

    return None


def _repair_missing_semdesc_description_object_closure(json_text):
    if not isinstance(json_text, str):
        return json_text

    pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]+)},[ \t]*(?P<newline>\r?\n)'
        r'(?P=indent)"comprehensive_description"[ \t]*:',
    )

    def add_missing_closure(match):
        indent = match.group("indent")
        newline = match.group("newline")
        return (
            f"{indent}  }}{newline}"
            f"{indent}}},{newline}"
            f'{indent}"comprehensive_description":'
        )

    return pattern.sub(add_missing_closure, json_text, count=1)


def parse_vlm_json_output_relaxed(raw_text, strict_parser):
    try:
        return strict_parser(raw_text)
    except Exception as strict_error:
        json_text = strip_markdown_json_fence(raw_text)
        try:
            return json.loads(json_text, strict=False)
        except Exception:
            repaired_json_text = _repair_json_string_backslashes(json_text)
            try:
                return json.loads(repaired_json_text, strict=False)
            except Exception:
                repair_sources = [json_text]
                if repaired_json_text != json_text:
                    repair_sources.append(repaired_json_text)

                for repair_source in repair_sources:
                    schema_repaired_json_text = (
                        _repair_missing_semdesc_description_object_closure(
                            repair_source
                        )
                    )
                    if schema_repaired_json_text == repair_source:
                        continue

                    parsed = _load_semdesc_json_candidate(schema_repaired_json_text)
                    if parsed is not None:
                        return parsed

                raise strict_error
