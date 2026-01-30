# 调用 OpenRouter API 代码，核心函数为 `ask_gpt`，使用示例:
# result = ask_gpt(
#     model_name="openai/gpt-5",
#     prompt="Describe what you see in these images.",
#     image=[pil_image, local_image_path, web_image_url]
# )
# print(json.dumps(result, ensure_ascii=False, indent=2))

import base64
import io
import json
import logging
import os
import time
import uuid

import requests
from PIL import Image


class Config:
    # OpenRouter Chat Completions
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    # 从 .env 读取（不额外引入依赖：简单解析 KEY=VALUE）
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

    ACCESS_KEY = os.getenv("OPENROUTER_API_KEY", "")
    DEFAULT_MODEL = "openai/gpt-5"

    # 重试策略
    MAX_RETRIES = 3
    WAIT_SECONDS_429 = 5  # 遇到 429 暂停秒数
    WAIT_SECONDS_ERROR = 2  # 其他错误暂停秒数


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """将 PIL Image 对象转换为 Base64 字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format=fmt)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_str}"


def _file_to_base64(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    if not ext:
        ext = "jpeg"
    if ext == "jpg":
        ext = "jpeg"

    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{ext};base64,{encoded_string}"


def _process_single_image(image_input: str | Image.Image) -> dict[str, str]:
    payload = {"type": "input_image"}

    try:
        if isinstance(image_input, Image.Image):
            payload["image_url"] = _image_to_base64(image_input)

        elif isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                payload["image_url"] = image_input
            elif os.path.exists(image_input):
                payload["image_url"] = _file_to_base64(image_input)
            else:
                raise ValueError(f"Invalid image path or URL: {image_input}")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        return payload
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise e


def ask_gpt(
    prompt: str,
    image: str | Image.Image | list[str | Image.Image] = None,
    model_name: str = Config.DEFAULT_MODEL
) -> dict[str, str]:
    """
    同步请求 GPT 模型（OpenRouter Chat Completions）。

    Args:
        model_name: 模型名称（如 openai/gpt-5）
        prompt: 问题文本
        image: 图片输入，支持 URL字符串 / 本地路径字符串 / PIL.Image对象，也支持以上类型的列表(多图)

    Returns:
        Dict:包含 "question" 和 "answer"
    """
    url = Config.BASE_URL
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {Config.ACCESS_KEY}",
        'X-TT-LOGID': str(uuid.uuid4()),

        # OpenRouter 推荐（可选）
        # 'HTTP-Referer': 'https://your-site.com',
        # 'X-Title': 'your-app-name',
    }

    # OpenAI/OpenRouter 多模态格式
    content_list = [{"type": "text", "text": prompt}]

    if image:
        if not isinstance(image, list):
            image = [image]

        for img_item in image:
            img_payload = _process_single_image(img_item)  # {"type":"input_image","image_url":"..."}
            content_list.append({
                "type": "image_url",
                "image_url": {"url": img_payload["image_url"]}
            })

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": content_list}
        ]
    }

    for attempt in range(Config.MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                res_json = response.json()
                try:
                    answer_text = ""
                    choices = res_json.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        answer_text = msg.get("content", "") or ""
                    return {"question": prompt, "answer": answer_text}
                except Exception as parse_error:
                    logger.error(f"Response parsing error: {parse_error}")
                    return {"question": prompt, "answer": f"Error parsing response: {parse_error}"}

            elif response.status_code == 429:
                logger.warning(
                    f"Status 429: Too Many Requests. Sleeping {Config.WAIT_SECONDS_429}s... "
                    f"(Attempt {attempt+1}/{Config.MAX_RETRIES})"
                )
                time.sleep(Config.WAIT_SECONDS_429)
                continue

            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                if 500 <= response.status_code < 600:
                    time.sleep(Config.WAIT_SECONDS_ERROR)
                    continue
                return {"question": prompt, "answer": f"HTTP Error {response.status_code}"}

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            time.sleep(Config.WAIT_SECONDS_ERROR)

    return {"question": prompt, "answer": "Request failed after max retries."}


if __name__ == "__main__":
    # PIL
    pil_image = Image.new('RGB', (100, 100), color='blue')
    local_image = "/mnt/bn/luoruipu-disk-2/guyukai/code/VideoGen/dataset/test/A2B_simple.png"

    # 网络图片
    web_image = "https://static.wixstatic.com/media/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg/v1/fill/w_585,h_405,al_c,q_80,enc_auto/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg"

    # 调用函数
    result = ask_gpt(
        model_name="openai/gpt-5",
        prompt="Describe what you see in these images.",
        image=[pil_image, local_image, web_image]
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
