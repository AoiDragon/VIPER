import os
import json
import time
import base64
import requests
import argparse
from tqdm import tqdm
from urllib.parse import urlparse

# 配置常量
API_KEY = os.getenv("DASHSCOPE_API_KEY")
SUBMIT_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis'
TASK_URL_TEMPLATE = 'https://dashscope.aliyuncs.com/api/v1/tasks/{}'

def get_base64_image(image_path):
    """读取本地图片并转换为Base64 Data URI"""
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return None
  
    ext = os.path.splitext(image_path)[1].lower().replace('.', '')
    if ext == 'jpg': ext = 'jpeg'
    if ext == 'png': ext = "png"

  
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{ext};base64,{encoded_string}"

def submit_task(model, prompt, image_path, resolution="720P"):
    """提交视频生成任务"""
    if not API_KEY:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")

    img_base64 = get_base64_image(image_path)
    if not img_base64:
        return None

    headers = {
        'X-DashScope-Async': 'enable',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": model,
        "input": {
            "prompt": prompt,
            "img_url": img_base64
        },
        "parameters": {
            "resolution": resolution,
            "prompt_extend": False,
            "audio": False,
            "duration": 5,
            "shot_type": "single"
        }
    }

    try:
        response = requests.post(SUBMIT_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if 'output' in result and 'task_id' in result['output']:
            return result['output']['task_id']
        else:
            print(f"[Submit Error] Unexpected response: {result}")
            return None
    except Exception as e:
        print(f"[Submit Error] {e}")
        return None

def check_task_status(task_id):
    """查询任务状态"""
    headers = {'Authorization': f'Bearer {API_KEY}'}
    try:
        url = TASK_URL_TEMPLATE.format(task_id)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[Check Error] Task {task_id}: {e}")
        return None

def download_video(url, save_path):
    """下载视频到本地"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[Download Error] {url} -> {save_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch Video Generation Script")
    parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_root', type=str, default='/mnt/bn/zhaoziwang/liyifan/code/Video-Reasoning-Bench/inference/results', help='Root directory for results')
    parser.add_argument('--model', type=str, default='wan2.6-i2v', help='Model name')
    parser.add_argument('--roll', type=int, default=1, help='Number of videos to generate per prompt')
    parser.add_argument('--resolution', type=str, default='720P', help='Video resolution')
  
    args = parser.parse_args()

    # 1. 准备路径和数据
    task_name = os.path.splitext(os.path.basename(args.input_json))[0] # e.g., obj_move
    output_dir = os.path.join(args.output_root, f"test_{args.model}")
    video_save_dir = os.path.join(output_dir, f"{task_name}_videos")
    output_json_path = os.path.join(output_dir, f"{task_name}.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_save_dir, exist_ok=True)

    print(f"Loading input: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 提交任务
    # 结构: pending_tasks = [ { "data_idx": 0, "roll_idx": 0, "task_id": "xxx" }, ... ]
    pending_tasks = []
  
    print(f"Starting submission for {len(data)} items (Roll={args.roll})...")
  
    # 初始化结果字段
    for item in data:
        item['server_url'] = []

    for idx, item in enumerate(tqdm(data, desc="Submitting Tasks")):
        prompt = item.get('prompt')
        image_path = item.get('image')
        item_id = item.get('id', f'item_{idx}')

        if not prompt or not image_path:
            print(f"Skipping item {idx}: Missing prompt or image")
            continue

        for r in range(args.roll):
            task_id = submit_task(args.model, prompt, image_path, args.resolution)
            if task_id:
                pending_tasks.append({
                    "data_idx": idx,
                    "roll_idx": r,
                    "task_id": task_id,
                    "item_id": item_id
                })
            else:
                print(f"Failed to submit task for item {item_id} roll {r}")
          
            # 简单的速率限制，防止QPS过高
            time.sleep(0.5)

    print(f"Submitted {len(pending_tasks)} tasks. Waiting for results...")

    # 3. 轮询状态
    # 简单的轮询逻辑：每隔一段时间检查所有未完成的任务
    # 优化：为了避免频繁请求，可以使用一个完成列表
    completed_count = 0
    total_tasks = len(pending_tasks)
  
    # 映射 task_id 到 pending_task 对象以便快速查找
    active_tasks = {t['task_id']: t for t in pending_tasks}

    while active_tasks:
        print(f"Polling... {len(active_tasks)} tasks remaining. (Total finished: {completed_count}/{total_tasks})")
      
        # 复制一份keys进行遍历，因为中途可能会删除字典项
        current_task_ids = list(active_tasks.keys())
      
        for task_id in current_task_ids:
            task_info = active_tasks[task_id]
            res = check_task_status(task_id)
          
            if not res:
                continue # 网络错误，稍后重试

            status = res.get('output', {}).get('task_status')
          
            if status == 'SUCCEEDED':
                video_url = res['output']['video_url']
                data_idx = task_info['data_idx']
                item_id = task_info['item_id']
                roll_idx = task_info['roll_idx']
              
                # 生成本地文件名
                file_name = f"{item_id}_{roll_idx}.mp4"
                local_path = os.path.abspath(os.path.join(video_save_dir, file_name))
              
                # 下载视频
                print(f"Downloading {item_id} (Roll {roll_idx})...")
                if download_video(video_url, local_path):
                    data[data_idx]['server_url'].append(local_path)
                else:
                    print(f"Failed to download video for task {task_id}")
              
                del active_tasks[task_id]
                completed_count += 1

            elif status in ['FAILED', 'CANCELED']:
                print(f"Task {task_id} failed or canceled.")
                # 可以选择记录错误信息到json
                del active_tasks[task_id]
                completed_count += 1
          
            # 如果是 PENDING 或 RUNNING，什么都不做，保留在 active_tasks 中

        if active_tasks:
            time.sleep(10) # 等待10秒再次轮询

    # 4. 保存结果
    print(f"Saving results to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
  
    print("Done.")

if __name__ == "__main__":
    main()
