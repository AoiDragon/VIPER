import os
import io
import re
import json
import time
import uuid
import base64
import logging
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from dotenv import load_dotenv


# ----------------------------
# Config & logging
# ----------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    # OpenRouter
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    default_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

    # Retry
    max_retries: int = 5
    wait_seconds_429: float = 5.0
    wait_seconds_error: float = 2.0

    # Request
    timeout_seconds: int = 120


CFG = AppConfig()


# ----------------------------
# IO helpers
# ----------------------------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


# ----------------------------
# Media helpers
# ----------------------------

def download_to_tempfile(url: str, suffix: str = ".mp4", timeout: int = 60) -> str:
    # Download remote file into a temp file and return its path.
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def extract_frames_from_video(video_path_or_url: str, fps: float = 1.0) -> List[Image.Image]:
    # Sample frames uniformly; always include first and last.
    temp_path: Optional[str] = None
    path = video_path_or_url

    if path.startswith(("http://", "https://")):
        temp_path = download_to_tempfile(path, suffix=".mp4")
        path = temp_path

    try:
        vr = VideoReader(path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path_or_url}")

        video_fps = max(vr.get_avg_fps(), 1e-6)
        duration = total_frames / video_fps
        target_n = max(2, int(round(duration * fps)))

        if target_n >= total_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, target_n).astype(int).tolist()
            indices[-1] = total_frames - 1
            indices = sorted(set(indices))

        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f.astype("uint8")) for f in frames]

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {temp_path}, err={e}")


def load_images(paths: List[str]) -> List[Image.Image]:
    # Load existing images only.
    images: List[Image.Image] = []
    for p in paths or []:
        if not p or not os.path.exists(p):
            logger.warning(f"Image not found: {p}")
            continue
        try:
            images.append(Image.open(p))
        except Exception as e:
            logger.warning(f"Failed to load image: {p}, err={e}")
    return images


def image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    # Convert PIL image to base64 data URL.
    if image.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode in ("RGBA", "LA"):
            bg.paste(image, mask=image.split()[-1])
        else:
            bg.paste(image)
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


# ----------------------------
# LLM response parsing
# ----------------------------

def extract_answer(response_text: str) -> Dict[str, Any]:
    # Extract <answer> JSON fields from the model output.
    m = re.search(r"\s*<answer>(.*?)</answer>", response_text, re.DOTALL | re.IGNORECASE)
    if not m:
        return {"valid": False, "decision": "", "process_consistency": "", "outcome_consistency": ""}

    answer = m.group(1).strip()

    # Strip fenced code blocks.
    answer = re.sub(r"^```json\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"^```\s*", "", answer).strip()
    answer = re.sub(r"\s*```$", "", answer).strip()

    # Prefer JSON.
    try:
        obj = json.loads(answer)
        decision = str(obj.get("decision", "")).lower()
        pc = str(obj.get("process_consistency", "")).lower()
        oc = str(obj.get("outcome_consistency", "")).lower()

        valid = {"correct", "incorrect"}
        if decision in valid and pc in valid and oc in valid:
            return {"valid": True, "decision": decision, "process_consistency": pc, "outcome_consistency": oc}
    except Exception:
        pass

    # Fallback: plain text.
    low = answer.lower()
    if "incorrect" in low:
        return {"valid": True, "decision": "incorrect", "process_consistency": "", "outcome_consistency": ""}
    if "correct" in low:
        return {"valid": True, "decision": "correct", "process_consistency": "", "outcome_consistency": ""}

    return {"valid": False, "decision": "", "process_consistency": "", "outcome_consistency": ""}


# ----------------------------
# OpenRouter client
# ----------------------------

def _require_api_key() -> None:
    if not CFG.api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env / environment.")


def ask_openrouter(content_list: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    # Call OpenRouter Chat Completions API with retry.
    _require_api_key()

    url = f"{CFG.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CFG.api_key}",
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4()),
        # Optional but recommended by OpenRouter:
        # "HTTP-Referer": "https://your-app.example",
        # "X-Title": "Video-Reasoning-Bench-Eval",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": 0.0,
    }

    for attempt in range(CFG.max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=CFG.timeout_seconds)

            if resp.status_code == 200:
                data = resp.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                extracted = extract_answer(text)
                if extracted["valid"]:
                    return {
                        "success": True,
                        "raw_answer": text,
                        "decision": extracted["decision"],
                        "process_consistency": extracted["process_consistency"],
                        "outcome_consistency": extracted["outcome_consistency"],
                    }

                logger.warning(
                    f"Invalid response format (attempt {attempt+1}/{CFG.max_retries})."
                )
                if attempt < CFG.max_retries - 1:
                    time.sleep(CFG.wait_seconds_error)
                    continue

                return {
                    "success": False,
                    "raw_answer": text,
                    "decision": "",
                    "process_consistency": "",
                    "outcome_consistency": "",
                    "error": "Invalid response format after max retries",
                }

            if resp.status_code == 429:
                logger.warning(
                    f"429 rate limit. Sleep {CFG.wait_seconds_429}s (attempt {attempt+1}/{CFG.max_retries})"
                )
                time.sleep(CFG.wait_seconds_429)
                continue

            # Retry on 5xx; fail fast on others.
            if 500 <= resp.status_code < 600:
                logger.error(f"Server error {resp.status_code}: {resp.text}")
                time.sleep(CFG.wait_seconds_error)
                continue

            return {
                "success": False,
                "raw_answer": "",
                "decision": "",
                "process_consistency": "",
                "outcome_consistency": "",
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }

        except requests.RequestException as e:
            logger.error(f"Network error: {e} (attempt {attempt+1}/{CFG.max_retries})")
            if attempt < CFG.max_retries - 1:
                time.sleep(CFG.wait_seconds_error)

    return {
        "success": False,
        "raw_answer": "",
        "decision": "",
        "process_consistency": "",
        "outcome_consistency": "",
        "error": "Request failed after max retries",
    }


# ----------------------------
# Prompt builder
# ----------------------------

def build_evaluation_content(
    system_prompt: str,
    domain_prompt: str,
    task_prompt: str,
    protocol: Union[str, List[str]],
    init_image: Optional[Image.Image],
    video_frames: List[Image.Image],
    reference_images: Optional[List[Image.Image]] = None,
    reference_text: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    # Build OpenAI-style multimodal content list.
    content: List[Dict[str, Any]] = []

    base_text = (
        f"{system_prompt}\n\n{domain_prompt}\n\n"
        f"The task prompt:\n{task_prompt}\n\n"
        f"The initial image:\n"
    )
    content.append({"type": "text", "text": base_text})

    if init_image is not None:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(init_image)}})

    protocol_text = "\n".join(protocol) if isinstance(protocol, list) else str(protocol or "")
    content.append({"type": "text", "text": f"Process constraints:\n{protocol_text}\n"})

    content.append({"type": "text", "text": "The generated video:\n"})
    for fr in video_frames:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(fr)}})

    if reference_images:
        intro = (
            "To assist your decision, here are reference frames sampled from a correct video:\n"
            if len(reference_images) > 1
            else "To assist your decision, here is a correct target frame:\n"
        )
        content.append({"type": "text", "text": intro})
        for img in reference_images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})

    if reference_text:
        content.append({"type": "text", "text": "To assist your decision, here is an example correct text reasoning process:\n"})
        for t in reference_text:
            content.append({"type": "text", "text": t})

    content.append({
        "type": "text",
        "text": (
            "Return your reasoning inside and the final JSON inside "
            "<answer>...</answer>. The JSON must include: decision, process_consistency, outcome_consistency. "
            "Each value must be either 'correct' or 'incorrect'."
        ),
    })

    return content


# ----------------------------
# Task pipeline
# ----------------------------

def prepare_task(
    video_url: str,
    video_idx: int,
    item: Dict[str, Any],
    system_prompt: str,
    domain_prompt: str,
    init_image: Optional[Image.Image],
    reference_images: List[Image.Image],
    reference_text: List[str],
    fps: float,
) -> Dict[str, Any]:
    # Extract frames and build content for one evaluation.
    try:
        frames = extract_frames_from_video(video_url, fps=fps)
        content_list = build_evaluation_content(
            system_prompt=system_prompt,
            domain_prompt=domain_prompt,
            task_prompt=item["prompt"],
            protocol=item.get("protocol", ""),
            init_image=init_image,
            video_frames=frames,
            reference_images=reference_images,
            reference_text=reference_text,
        )
        return {
            "success": True,
            "content_list": content_list,
            "item_id": item.get("id", "unknown"),
            "domain": item["domain"],
            "server_url": video_url,
            "video_idx": video_idx,
            "task_prompt": item["prompt"],
            "protocol": item.get("protocol", ""),
            "error": "",
        }
    except Exception as e:
        logger.error(f"Failed to prepare task: url={video_url}, err={e}")
        return {
            "success": False,
            "content_list": None,
            "item_id": item.get("id", "unknown"),
            "domain": item.get("domain", "unknown"),
            "server_url": video_url,
            "video_idx": video_idx,
            "task_prompt": item.get("prompt", ""),
            "protocol": item.get("protocol", ""),
            "error": str(e),
        }


def evaluate_task(task: Dict[str, Any], model: str) -> Dict[str, Any]:
    # Evaluate one prepared task via OpenRouter.
    if not task.get("success") or not task.get("content_list"):
        return {
            **{k: task.get(k) for k in ["item_id", "domain", "server_url", "video_idx", "task_prompt", "protocol"]},
            "raw_answer": "",
            "decision": "",
            "process_consistency": "",
            "outcome_consistency": "",
            "success": False,
            "error": task.get("error", "Task preparation failed"),
        }

    result = ask_openrouter(task["content_list"], model=model)
    return {
        **{k: task.get(k) for k in ["item_id", "domain", "server_url", "video_idx", "task_prompt", "protocol"]},
        "raw_answer": result.get("raw_answer", ""),
        "decision": result.get("decision", ""),
        "process_consistency": result.get("process_consistency", ""),
        "outcome_consistency": result.get("outcome_consistency", ""),
        "success": bool(result.get("success")),
        "error": result.get("error", ""),
    }


# ----------------------------
# Main
# ----------------------------

def list_input_files(data_path: str, file_name: Optional[str]) -> List[str]:
    # Resolve json input files.
    if os.path.isfile(data_path):
        return [data_path]

    if os.path.isdir(data_path):
        if file_name:
            candidate = os.path.join(data_path, f"{file_name}.json")
            if os.path.isfile(candidate) and candidate.endswith(".json"):
                return [candidate]
            raise FileNotFoundError(f"Specified file not found: {candidate}")

        return [
            os.path.join(data_path, f)
            for f in sorted(os.listdir(data_path))
            if f.endswith(".json")
        ]

    raise FileNotFoundError(f"Invalid data_path: {data_path}")


def load_init_image(path: Optional[str]) -> Optional[Image.Image]:
    # Load init image if exists.
    if not path or not os.path.exists(path):
        return None
    try:
        return Image.open(path)
    except Exception as e:
        logger.error(f"Failed to load init image: {path}, err={e}")
        return None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--system_prompt_path", type=str, required=True)
    parser.add_argument("--domain_prompt_root", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="JSON file or directory")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default=CFG.default_model)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--k", type=int, default=None, help="Evaluate top-k videos per item")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    system_prompt = read_text(args.system_prompt_path)
    input_files = list_input_files(args.data_path, args.file_name)

    if not input_files:
        raise RuntimeError(f"No JSON files found in {args.data_path}")

    if args.output_path is None:
        base_dir = os.path.dirname(args.data_path) if os.path.isfile(args.data_path) else args.data_path
        args.output_path = base_dir.replace("inference", "evaluation")
    os.makedirs(args.output_path, exist_ok=True)

    logger.info(f"Starting evaluation: {len(input_files)} file(s)")
    logger.info(f"Output directory: {args.output_path}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Model: {args.model_name}")

    for file_idx, input_file in enumerate(input_files, start=1):
        try:
            dataset = read_json(input_file)
            if isinstance(dataset, dict):
                dataset = [dataset]

            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            model_suffix = args.model_name.replace("/", "_").replace("\\", "_")
            fps_info = f"fps@{args.fps}"
            pass_info = f"pass@{args.k}" if args.k else "pass@all"
            output_filename = f"{input_basename}_{fps_info}_{pass_info}_{model_suffix}.json"
            output_path = os.path.join(args.output_path, output_filename)

            # Resume cache: only keep successful ones.
            existing_success: Dict[tuple, Dict[str, Any]] = {}
            if args.resume and os.path.exists(output_path):
                try:
                    prev = read_json(output_path)
                    for r in prev:
                        if r.get("success"):
                            existing_success[(r.get("item_id"), r.get("video_idx"))] = r
                    logger.info(f"Resume: loaded {len(existing_success)} successful results")
                except Exception as e:
                    logger.warning(f"Resume: failed to load {output_path}, err={e}")

            total_videos = sum(len(item.get("server_url", [])) for item in dataset)
            logger.info(f"[{file_idx}/{len(input_files)}] {input_basename}: {len(dataset)} items, {total_videos} videos")

            # Stage 1: prepare tasks (frames + content).
            tasks: List[Dict[str, Any]] = []
            for item_idx, item in enumerate(dataset):
                item_id = item.get("id", f"item_{item_idx}")
                domain = item["domain"]

                domain_prompt_path = os.path.join(args.domain_prompt_root, f"{domain}.txt")
                domain_prompt = read_text(domain_prompt_path)

                init_image = load_init_image(item.get("image"))
                reference_images = load_images(item.get("reference_frames", []))
                reference_text = item.get("reference_text", [])

                urls = item.get("server_url", [])
                if args.k and args.k > 0:
                    urls = urls[: args.k]

                if not urls:
                    key = (item_id, 0)
                    if not (args.resume and key in existing_success):
                        tasks.append({
                            "success": False,
                            "content_list": None,
                            "item_id": item_id,
                            "domain": domain,
                            "server_url": "",
                            "video_idx": 0,
                            "task_prompt": item.get("prompt", ""),
                            "protocol": item.get("protocol", ""),
                            "error": "No videos available for this item",
                        })
                    continue

                for video_idx, url in enumerate(urls):
                    key = (item_id, video_idx)
                    if args.resume and key in existing_success:
                        continue
                    tasks.append(prepare_task(
                        video_url=url,
                        video_idx=video_idx,
                        item=item,
                        system_prompt=system_prompt,
                        domain_prompt=domain_prompt,
                        init_image=init_image,
                        reference_images=reference_images,
                        reference_text=reference_text,
                        fps=args.fps,
                    ))

            logger.info(f"Prepared {len(tasks)} records")
            if args.resume and len(tasks) == 0:
                logger.info("All evaluations already completed, skipping.\n")
                continue

            # Stage 2: run model calls.
            new_results: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_map = {executor.submit(evaluate_task, t, args.model_name): t for t in tasks}
                with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                    for fut in as_completed(future_map):
                        try:
                            new_results.append(fut.result())
                        except Exception as e:
                            t = future_map[fut]
                            logger.error(f"Task crashed: {e}")
                            new_results.append({
                                **{k: t.get(k) for k in ["item_id", "domain", "server_url", "video_idx", "task_prompt", "protocol"]},
                                "raw_answer": "",
                                "decision": "",
                                "process_consistency": "",
                                "outcome_consistency": "",
                                "success": False,
                                "error": str(e),
                            })
                        finally:
                            pbar.update(1)

            all_results = list(existing_success.values()) + new_results if (args.resume and existing_success) else new_results
            write_json(output_path, all_results)

            ok = sum(1 for r in all_results if r.get("success"))
            logger.info(f"  ✓ Completed: {ok}/{len(all_results)} successful")
            logger.info(f"  ✓ Saved to {output_filename}\n")

        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}\n")

    logger.info(f"Evaluation completed: {len(input_files)} file(s) processed")


if __name__ == "__main__":
    main()
