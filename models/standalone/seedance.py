import os
import json
import base64
import asyncio
import aiohttp
import time
import argparse
from tqdm.asyncio import tqdm

# ================= 配置区域 =================
API_KEY = os.getenv("ARK_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "doubao-seedance-1-5-pro-251215"

CREATE_TASK_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
GET_TASK_URL_TEMPLATE = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{}"

PROMPT_SUFFIX = "--rs 480p --rt 16:9 --dur 5 --fps 24 --wm false --cf true"
CONCURRENCY_LIMIT = 5
# ===========================================

async def encode_image_to_base64(image_path):
    """读取本地图片并转换为Base64 Data URI"""
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return None

    ext = os.path.splitext(image_path)[1].lower().replace('.', '')
    if ext == 'jpg': ext = 'jpeg'

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{ext};base64,{encoded_string}"

async def submit_task(session, prompt, image_base64):
    """提交生成任务"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "content": [
            {"type": "text", "text": prompt + PROMPT_SUFFIX},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ],
        "generate_audionew": False
    }

    try:
        async with session.post(CREATE_TASK_URL, json=payload, headers=headers) as response:
            resp_json = await response.json()
            if "id" in resp_json:
                return resp_json["id"]
            else:
                print(f"[Submit Failed] {resp_json}")
                return None
    except Exception as e:
        print(f"[Submit Error] {e}")
        return None

async def poll_result(session, task_id):
    """轮询任务结果"""
    url = GET_TASK_URL_TEMPLATE.format(task_id)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    await asyncio.sleep(30)

    while True:
        try:
            async with session.get(url, headers=headers) as response:
                data = await response.json()
                status = data.get("status")

                if status == "succeeded":
                    return data.get("content", {}).get("video_url")
                elif status == "failed":
                    print(f"[Task Failed] ID: {task_id}")
                    return None
                elif status in ["running", "queued"]:
                    await asyncio.sleep(10)
                else:
                    print(f"[Unknown Status] {status} for {task_id}")
                    return None
        except Exception as e:
            print(f"[Poll Error] {e}")
            await asyncio.sleep(5)

async def download_video(session, url, save_path):
    """下载视频到本地"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(await response.read())
                return os.path.abspath(save_path)
            else:
                print(f"[Download Failed] Status {response.status} for {url}")
                return None
    except Exception as e:
        print(f"[Download Error] {e}")
        return None

async def process_single_item(sem, session, item, output_video_dir, roll_count):
    """处理单个JSON条目，包含多次roll"""
    result_item = item.copy()
    result_item["server_urls"] = []  # only store downloaded absolute local paths

    image_path = item.get("image")
    prompt = item.get("prompt")
    item_id = item.get("id")

    image_b64 = await encode_image_to_base64(image_path)
    if not image_b64:
        return result_item

    tasks = []
    for i in range(roll_count):
        async with sem:
            task_id = await submit_task(session, prompt, image_b64)
            if task_id:
                tasks.append((i, task_id))

    for i, task_id in tasks:
        video_url = await poll_result(session, task_id)

        if video_url:
            file_name = f"{item_id}_{i}.mp4"
            save_path = os.path.join(output_video_dir, file_name)
            local_path = await download_video(session, video_url, save_path)
            if local_path:
                result_item["server_urls"].append(local_path)
            else:
                result_item["server_urls"].append("DOWNLOAD_FAILED")
        else:
            result_item["server_urls"].append("GENERATION_FAILED")

    return result_item

async def main(args):
    global MODEL_NAME
    MODEL_NAME = args.model

    input_path = args.input_json
    task_name = os.path.splitext(os.path.basename(input_path))[0]

    base_output_dir = os.path.join(args.output_root, f"test_{MODEL_NAME}")
    output_json_path = os.path.join(base_output_dir, f"{task_name}.json")
    output_video_dir = os.path.join(base_output_dir, "videos", task_name)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output JSON: {output_json_path}")
    print(f"Output Videos: {output_video_dir}")
    print(f"Model: {MODEL_NAME}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_single_item(semaphore, session, item, output_video_dir, args.roll)
            for item in data
        ]

        for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing Tasks"):
            res = await f
            results.append(res)

    results.sort(key=lambda x: x['id'])

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Done! Results saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Video Generation Script")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for outputs")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--roll", type=int, default=1, help="Number of videos to generate per prompt")

    args = parser.parse_args()

    if not API_KEY or "YOUR_API_KEY" in API_KEY:
        print("Please set ARK_API_KEY environment variable.")
    else:
        asyncio.run(main(args))
