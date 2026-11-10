import os
import json
import yaml
import base64
from pathlib import Path



def load_prompt(prompt_path, prompt_name):
    prompt_path = Path(__file__).parent.parent / prompt_path

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    return prompts[prompt_name]


def read_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_json(data, save_path, format="json", ensure_ascii=False, indent=2, output_info=True):
    """
    将由字典组成的数据保存为 JSON 或 JSONL 文件。

    参数:
        data (list[dict] | dict): 要保存的数据。
        save_path (str): 文件保存路径。
        format (str): 保存格式，支持 "json" 或 "jsonl"。
        ensure_ascii (bool): 是否转义非 ASCII 字符，中文内容建议设为 False。
        indent (int): JSON 缩进空格数，仅对 JSON 格式有效。
        output_info (bool): 是否在保存后打印文件路径。
    """
    save_path = str(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if format.lower() == "jsonl":
        with open(save_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
    else:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)

    if output_info:
        print(f"[{format.upper()}] Save to : {save_path}")


def prepare_image(image: str) -> str:
    # 如果输入是本地图像路径，则转换为 base64 data URL。
    if not os.path.exists(image):
        return image

    with open(image, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # 根据文件后缀判断 MIME 类型。
    ext = os.path.splitext(image)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(
            f"Unsupported image format: {ext}"
        )

    return f"data:{mime};base64,{image_base64}"
