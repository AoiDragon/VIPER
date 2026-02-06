import requests
import json
import time
import uuid
import base64
import io
import os
import logging
import numpy as np
from typing import List, Union, Dict
from PIL import Image
from argparse import ArgumentParser
from decord import VideoReader, cpu
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    BASE_URL = "sample_url"
    ACCESS_KEY = "Sample_key"
    DEFAULT_MODEL = "gpt-4o-2024-11-20"
    
    # 重试策略
    MAX_RETRIES = 5
    WAIT_SECONDS_429 = 5  # 遇到 429 暂停秒数
    WAIT_SECONDS_ERROR = 2 # 其他错误暂停秒数

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def read_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_frames_from_video(video_path: str, fps: float = 1.0) -> List[Image.Image]:
    """
    从视频中按照指定fps采样帧，必须包含第一帧和最后一帧
    
    Args:
        video_path: 视频文件路径或URL
        fps: 采样帧率 (默认1.0)
    
    Returns:
        List[Image.Image]: PIL Image对象列表
    """
    # 如果是URL，先下载视频
    temp_file_path = None
    if video_path.startswith("http://") or video_path.startswith("https://"):
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        try:
            response = requests.get(video_path, timeout=60)
            response.raise_for_status()
            temp_file.write(response.content)
            temp_file.close()
            temp_file_path = temp_file.name
            video_path = temp_file_path
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
    
    try:
        # 使用decord读取视频
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # 根据fps计算采样的帧数
        video_duration = total_frames / video_fps  # 视频总时长（秒）
        num_frames = max(2, round(video_duration * fps))  # 至少2帧（首尾），四舍五入
        
        # 计算采样的帧索引
        if num_frames >= total_frames:
            # 如果请求的帧数>=总帧数，返回所有帧
            frame_indices = list(range(total_frames))
        else:
            # 均匀采样，确保包含第一帧和最后一帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
            # 确保包含最后一帧（linspace可能因为int截断而丢失）
            if frame_indices[-1] != total_frames - 1:
                frame_indices[-1] = total_frames - 1
            # 去重并保持顺序
            frame_indices = sorted(set(frame_indices))
        
        # 批量读取帧
        frames_array = vr.get_batch(frame_indices).asnumpy()
        
        # 转换为PIL Image对象
        frames = []
        for frame in frames_array:
            # decord读取的是RGB格式，直接转换为PIL Image
            pil_image = Image.fromarray(frame.astype('uint8'))
            frames.append(pil_image)
        
    except Exception as e:
        logger.error(f"Failed to extract frames: {e}")
        raise e
    finally:
        # 如果是临时文件，删除它
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
    
    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    return frames


def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """将 PIL Image 对象转换为 Base64 字符串"""
    # 如果图片是 RGBA 或 LA 模式，转换为 RGB
    if image.mode in ('RGBA', 'LA', 'P'):
        # 创建白色背景
        background = Image.new('RGB', image.size, (255, 255, 255))
        # 如果有 alpha 通道，使用它作为 mask
        if image.mode == 'RGBA' or image.mode == 'LA':
            background.paste(image, mask=image.split()[-1])  # 使用 alpha 通道
        else:
            background.paste(image)
        image = background
    elif image.mode != 'RGB':
        # 其他模式直接转换为 RGB
        image = image.convert('RGB')
    
    buffered = io.BytesIO()
    image.save(buffered, format=fmt)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_str}"


def extract_answer_from_response(response_text: str) -> Dict[str, str]:
    """
    从GPT响应中提取<think>和<answer>标签内容，支持JSON格式。
    
    Args:
        response_text: GPT的完整响应文本
    
    Returns:
        Dict: 包含 "decision", "process_consistency", "outcome_consistency", "valid" 的字典
    """
    import re
    
    # 一行匹配：必须同时包含<think>...</think>和<answer>...</answer>
    match = re.search(r'<think>.*?</think>\s*<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        return {"decision": "", "process_consistency": "", "outcome_consistency": "", "valid": False}
    
    answer_content = match.group(1).strip()
    
    # 尝试解析JSON格式
    try:
        # 清理可能的代码块标记
        answer_content = answer_content.strip()
        if answer_content.startswith('```json'):
            answer_content = answer_content[7:]
        if answer_content.startswith('```'):
            answer_content = answer_content[3:]
        if answer_content.endswith('```'):
            answer_content = answer_content[:-3]
        answer_content = answer_content.strip()
        
        # 解析JSON
        answer_json = json.loads(answer_content)
        
        # 提取字段并转换为小写
        process_consistency = answer_json.get("process_consistency", "").lower()
        outcome_consistency = answer_json.get("outcome_consistency", "").lower()
        decision = answer_json.get("decision", "").lower()
        
        # 验证格式
        valid_values = {"correct", "incorrect"}
        if (process_consistency in valid_values and 
            outcome_consistency in valid_values and 
            decision in valid_values):
            return {
                "decision": decision,
                "process_consistency": process_consistency,
                "outcome_consistency": outcome_consistency,
                "valid": True
            }
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    
    # 如果JSON解析失败，尝试旧格式（纯文本）
    answer_lower = answer_content.lower()
    if 'incorrect' in answer_lower:
        return {"decision": "incorrect", "process_consistency": "", "outcome_consistency": "", "valid": True}
    elif 'correct' in answer_lower:
        return {"decision": "correct", "process_consistency": "", "outcome_consistency": "", "valid": True}
    
    return {"decision": "", "process_consistency": "", "outcome_consistency": "", "valid": False}


def ask_gpt(
    content_list: List[Dict],
    model_name: str = Config.DEFAULT_MODEL
) -> Dict[str, str]:
    """
    使用自定义 content_list 请求 GPT 模型，支持文字和图片交错输入。
    包含格式验证和重试机制。
    
    Args:
        content_list: 内容列表，包含文字和图片
        model_name: 模型名称
    
    Returns:
        Dict: 包含 "raw_answer", "decision", "success" 的字典
    """
    # 准备请求头
    url = f"{Config.BASE_URL}?ak={Config.ACCESS_KEY}"
    headers = {
        'Content-Type': 'application/json',
        'X-TT-LOGID': str(uuid.uuid4())
    }
    
    # 构造完整 Payload
    payload = {
        "model": model_name,
        "input": [
            {
                "role": "user",
                "content": content_list
            }
        ]
    }

    # 发送请求 (包含重试逻辑)
    for attempt in range(Config.MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # --- 场景 A: 成功 ---
            if response.status_code == 200:
                res_json = response.json()
                
                # 解析返回结果，提取 answer
                try:
                    output_data = res_json.get("output", {})
                    answer_text = ""
                    if output_data:
                        first_content = output_data[0].get("content", [])
                        if first_content:
                            answer_text = first_content[0].get("text", "")
                    
                    # 提取和验证答案格式
                    extracted = extract_answer_from_response(answer_text)
                    
                    if extracted["valid"]:
                        # 格式正确，返回成功
                        return {
                            "raw_answer": answer_text,
                            "decision": extracted["decision"],
                            "process_consistency": extracted.get("process_consistency", ""),
                            "outcome_consistency": extracted.get("outcome_consistency", ""),
                            "success": True
                        }
                    else:
                        # 格式不正确，记录日志并重试
                        logger.warning(f"Invalid response format (attempt {attempt+1}/{Config.MAX_RETRIES}). Missing <think>/<answer> tags.")
                        if attempt < Config.MAX_RETRIES - 1:
                            time.sleep(Config.WAIT_SECONDS_ERROR)
                            continue
                        else:
                            # 最后一次尝试也失败，返回格式错误
                            return {
                                "raw_answer": answer_text,
                                "decision": "",
                                "process_consistency": "",
                                "outcome_consistency": "",
                                "success": False,
                                "error": "Invalid response format after max retries"
                            }
                    
                except Exception as parse_error:
                    logger.error(f"Response parsing error: {parse_error}")
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(Config.WAIT_SECONDS_ERROR)
                        continue
                    return {
                        "raw_answer": "",
                        "decision": "",
                        "process_consistency": "",
                        "outcome_consistency": "",
                        "success": False,
                        "error": f"Error parsing response: {parse_error}"
                    }

            # --- 场景 B: 429 限流 ---
            elif response.status_code == 429:
                logger.warning(f"Status 429: Too Many Requests. Sleeping {Config.WAIT_SECONDS_429}s... (Attempt {attempt+1}/{Config.MAX_RETRIES})")
                time.sleep(Config.WAIT_SECONDS_429)
                continue
            
            # --- 场景 C: 其他错误 ---
            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                # 5xx 错误稍微等一下再试
                if 500 <= response.status_code < 600:
                    time.sleep(Config.WAIT_SECONDS_ERROR)
                    continue
                return {
                    "raw_answer": "",
                    "decision": "",
                    "process_consistency": "",
                    "outcome_consistency": "",
                    "success": False,
                    "error": f"HTTP Error {response.status_code}: {response.text}"
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}, attempt {attempt+1}/{Config.MAX_RETRIES}")
            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.WAIT_SECONDS_ERROR)
            continue

    # 如果循环结束还没有返回，说明失败
    return {
        "raw_answer": "",
        "decision": "",
        "process_consistency": "",
        "outcome_consistency": "",
        "success": False,
        "error": "Request failed after max retries."
    }


def load_images(image_paths: List[str]) -> List[Image.Image]:
    """
    加载多个图片文件。
    
    Args:
        image_paths: 图片路径列表
    
    Returns:
        List[Image.Image]: 成功加载的PIL Image对象列表
    """
    images = []
    for path in image_paths:
        try:
            if os.path.exists(path):
                images.append(Image.open(path))
            else:
                logger.warning(f"Image not found: {path}")
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
    return images


def build_evaluation_content_list(
    system_prompt: str,
    domain_prompt: str,
    task_prompt: str,
    protocol: Union[str, List[str]],
    init_image: Image.Image,
    video_frames: List[Image.Image],
    reference_images: List[Image.Image] = None,
    reference_text: List[str] = None
) -> List[Dict]:
    """
    构建交错的评测内容列表（文字和图片交错）。
    
    Args:
        system_prompt: 系统提示词
        domain_prompt: 领域提示词
        task_prompt: 任务提示词
        protocol: 流程协议（字符串或列表）
        init_image: 初始图片
        video_frames: 视频帧列表
        reference_images: 参考图片列表（可选）
    
    Returns:
        List[Dict]: 交错的内容列表
    """
    content_list = []
    
    # 1. 添加基础说明文字
    base_text = f"{system_prompt}\n\n{domain_prompt}\n\nThe task prompt:\n{task_prompt}\n\nThe initial image:\n"
    content_list.append({"type": "input_text", "text": base_text})
    
    # 2. 添加初始图片
    if init_image:
        content_list.append({"type": "input_image", "image_url": _image_to_base64(init_image)})

    # 3. 流程限制
    if isinstance(protocol, list):
        protocol_text = '\n'.join(protocol)
    else:
        protocol_text = protocol
    constraints = f"Process constraints:\n{protocol_text}\n"
    content_list.append({"type": "input_text", "text": constraints})

    # 3. 添加视频帧说明和图片
    video_text = f"\nThe generated video:\n"
    content_list.append({"type": "input_text", "text": video_text})
    
    for frame in video_frames:
        content_list.append({"type": "input_image", "image_url": _image_to_base64(frame)})

    # 4. 添加参考
    if reference_images:
        if len(reference_images) > 1:
            reference_intro = f"\n To assist your decision, here are reference frames sampled from a correct video:\n"
            content_list.append({"type": "input_text", "text": reference_intro})
        else:
            reference_intro = f"\n To assist your decision, here is a correct target frame:\n"
            content_list.append({"type": "input_text", "text": reference_intro})
        
        for ref_image in reference_images:
            content_list.append({"type": "input_image", "image_url": _image_to_base64(ref_image)})
    
    if reference_text and len(reference_text) > 0:
        reference_intro = "\n To assist your decision, here is an example correct text reasoning process:\n"
        content_list.append({"type": "input_text", "text": reference_intro})
        for ref_text in reference_text:
            content_list.append({"type": "input_text", "text": ref_text})

    final_instruction = "Please first think step-by-step within <think> tags, then provide your final decision within <answer> tags."
    content_list.append({"type": "input_text", "text": final_instruction})
    
    return content_list


def prepare_evaluation_task(
    s_url: str,
    video_idx: int,
    item_data: Dict,
    system_prompt: str,
    domain_prompt: str,
    init_image: Image.Image,
    reference_images: List[Image.Image],
    reference_text: List[str],
    fps: float
) -> Dict:
    """
    准备单个评测任务的所有数据（提取视频帧并构造content list）。
    
    Args:
        video_url: 视频URL
        video_idx: 视频索引
        item_data: 数据项（包含id, domain, prompt, protocol等）
        system_prompt: 系统提示词
        domain_prompt: 领域提示词
        init_image: 初始图片
        reference_images: 参考图片列表
        fps: 采样帧率
    
    Returns:
        Dict: 包含content_list和元数据的任务字典
    """
    try:
        # 从视频中提取帧
        video_frames = extract_frames_from_video(s_url, fps=fps)
        
        # 构建评测内容列表
        content_list = build_evaluation_content_list(
            system_prompt=system_prompt,
            domain_prompt=domain_prompt,
            task_prompt=item_data['prompt'],
            protocol=item_data.get('protocol', ''),
            init_image=init_image,
            video_frames=video_frames,
            reference_images=reference_images,
            reference_text=reference_text
        )
        
        return {
            "content_list": content_list,
            "item_id": item_data.get('id', 'unknown'),
            "domain": item_data['domain'],
            "server_url": s_url,
            "video_idx": video_idx,
            "task_prompt": item_data['prompt'],
            "protocol": item_data.get('protocol', ''),
            "success": True,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare task for video {s_url}: {e}")
        return {
            "content_list": None,
            "item_id": item_data.get('id', 'unknown'),
            "domain": item_data['domain'],
            "server_url": s_url,
            "video_idx": video_idx,
            "task_prompt": item_data['prompt'],
            "protocol": item_data.get('protocol', ''),
            "success": False,
            "error": str(e)
        }


def evaluate_task_with_gpt(task: Dict, model_name: str) -> Dict:
    """
    使用GPT评测单个任务。
    
    Args:
        task: 包含content_list和元数据的任务字典
        model_name: 模型名称
    
    Returns:
        Dict: 评测结果
    """
    # 如果任务准备失败，直接返回失败结果
    if not task["success"] or task["content_list"] is None:
        return {
            "item_id": task["item_id"],
            "domain": task["domain"],
            "server_url": task["server_url"],
            "video_idx": task["video_idx"],
            "raw_answer": "",
            "decision": "",
            "process_consistency": "",
            "outcome_consistency": "",
            "success": False,
            "task_prompt": task["task_prompt"],
            "protocol": task["protocol"],
            "error": task.get("error", "Task preparation failed")
        }
    
    try:
        # 调用GPT-4o进行评测
        result = ask_gpt(task["content_list"], model_name=model_name)
        
        return {
            "item_id": task["item_id"],
            "domain": task["domain"],
            "server_url": task["server_url"],
            "video_idx": task["video_idx"],
            "raw_answer": result.get("raw_answer", ""),
            "decision": result.get("decision", ""),
            "process_consistency": result.get("process_consistency", ""),
            "outcome_consistency": result.get("outcome_consistency", ""),
            "success": result["success"],
            "task_prompt": task["task_prompt"],
            "protocol": task["protocol"],
            "error": result.get("error", "")
        }
        
    except Exception as e:
        logger.error(f"Error evaluating task: {e}")
        return {
            "item_id": task["item_id"],
            "domain": task["domain"],
            "server_url": task["server_url"],
            "video_idx": task["video_idx"],
            "raw_answer": "",
            "decision": "",
            "process_consistency": "",
            "outcome_consistency": "",
            "success": False,
            "task_prompt": task["task_prompt"],
            "protocol": task["protocol"],
            "error": str(e)
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--system_prompt_path", type=str, default="/mnt/bn/zhaoziwang/liyifan/code/Video-Reasoning-Bench/evaluation/prompt/system_prompt.txt")
    parser.add_argument("--domain_prompt_root", type=str, default="/mnt/bn/zhaoziwang/liyifan/code/Video-Reasoning-Bench/evaluation/prompt/domain_prompt/")
    parser.add_argument("--data_path", type=str, default="/mnt/bn/zhaoziwang/liyifan/code/Video-Reasoning-Bench/inference/results/test/", help="输入数据路径，可以是文件或目录")
    parser.add_argument("--output_path", type=str, default=None, help="输出目录路径（默认在data_path的evaluation对应目录）")
    parser.add_argument("--file_name", type=str, default=None, help="指定要评测的文件名（仅当data_path是目录时有效）")
    parser.add_argument("--fps", type=float, default=1.0, help="采样帧率")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20", help="使用的模型名称")
    parser.add_argument("--max_workers", type=int, default=10, help="多线程worker数量")
    parser.add_argument("--k", type=int, default=None, help="pass@k，每个item评测前k个视频（默认评测所有视频）")
    parser.add_argument("--resume", action="store_true", help="恢复模式：跳过已成功评测的记录，只评测失败或缺失的记录")
    args = parser.parse_args()
    
    # 读取系统提示词
    system_prompt = read_text(args.system_prompt_path)
    
    # 确定输入文件列表
    input_files = []
    if os.path.isfile(args.data_path):
        # 如果是单个文件
        input_files = [args.data_path]
    elif os.path.isdir(args.data_path):
        # 如果是目录
        if args.file_name:
            # 指定了文件名，只处理该文件
            target_file = os.path.join(args.data_path, args.file_name) + '.json'
            if os.path.isfile(target_file) and target_file.endswith('.json'):
                input_files = [target_file]
            else:
                logger.error(f"Specified file not found or not a JSON file: {target_file}")
                exit(1)
        else:
            # 未指定文件名，获取所有json文件
            input_files = [
                os.path.join(args.data_path, f) 
                for f in sorted(os.listdir(args.data_path)) 
                if f.endswith('.json')
            ]
    else:
        logger.error(f"Invalid data_path: {args.data_path}")
        exit(1)
    
    if not input_files:
        logger.error(f"No JSON files found in {args.data_path}")
        exit(1)
    
    # 确定输出目录
    if args.output_path is None:
        if os.path.isfile(args.data_path):
            base_dir = os.path.dirname(args.data_path)
        else:
            base_dir = args.data_path
        args.output_path = base_dir.replace("inference", "evaluation")
    
    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)
    
    logger.info(f"Starting evaluation: {len(input_files)} file(s) to process")
    logger.info(f"Output directory: {args.output_path}")
    logger.info(f"Max workers: {args.max_workers}\n")
    
    # 遍历每个输入文件
    for file_idx, input_file in enumerate(input_files):
        try:
            # 读取数据
            dataset = read_json(input_file)
            
            # 检查dataset格式：如果是单个字典，转换为列表
            if isinstance(dataset, dict):
                dataset = [dataset]
            
            # 生成输出文件名：输入文件名（不含扩展名）+ fps@n + pass@k + 模型名 + .json
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            model_suffix = args.model_name.replace('/', '_').replace('\\', '_')  # 处理模型名中的特殊字符
            
            # 构建文件名：包含 fps@n 和 pass@k 信息
            fps_info = f"fps@{args.fps}"
            if args.k is not None:
                pass_info = f"pass@{args.k}"
            else:
                pass_info = "pass@all"
            
            output_filename = f"{input_basename}_{fps_info}_{pass_info}_{model_suffix}.json"
            output_file_path = os.path.join(args.output_path, output_filename)
            
            # ========== Resume 模式: 读取已有结果 ==========
            existing_results = {}
            if args.resume and os.path.exists(output_file_path):
                try:
                    with open(output_file_path, 'r', encoding='utf-8') as f:
                        prev_results = json.load(f)
                    
                    # 建立索引：(item_id, video_idx) -> result
                    for result in prev_results:
                        if result.get('success', False):
                            key = (result.get('item_id'), result.get('video_idx'))
                            existing_results[key] = result
                    
                    logger.info(f"Resume mode: Found {len(existing_results)} successful evaluations from previous run")
                except Exception as e:
                    logger.warning(f"Failed to load previous results from {output_file_path}: {e}")
                    existing_results = {}
            
            total_videos = sum(len(item.get('server_url', [])) for item in dataset)
            
            logger.info(f"[{file_idx + 1}/{len(input_files)}] {input_basename}: {len(dataset)} items, {total_videos} videos")
            
            # ========== 阶段1: 准备所有任务 ==========
            all_tasks = []
            
            for item_idx, item in enumerate(dataset):
                domain = item['domain']
                domain_prompt_path = os.path.join(args.domain_prompt_root, f"{domain}.txt")
                domain_prompt = read_text(domain_prompt_path)
                
                item_id = item.get('id', f'item_{item_idx}')
                
                # 加载初始图片
                init_image = None
                if item.get('image') and os.path.exists(item['image']):
                    try:
                        init_image = Image.open(item['image'])
                    except Exception as e:
                        logger.error(f"Failed to load initial image for item {item_id}: {e}")
                
                # 加载参考图片
                reference_images = load_images(item['reference_frames'])
                reference_text = item["reference_text"]

                # 根据 k 值限制要评测的视频数量
                server_urls = item.get('server_url', [])
                
                if args.k is not None and args.k > 0:
                    # 只取前 k 个视频
                    server_urls = server_urls[:args.k]
                
                # 如果该 item 没有任何视频，创建一个占位记录
                if not server_urls:
                    # 检查是否已经有占位记录
                    task_key = (item_id, 0)
                    if not (args.resume and task_key in existing_results):
                        # 创建一个失败的占位任务（不需要实际评测）
                        placeholder_task = {
                            "content_list": None,
                            "item_id": item_id,
                            "domain": item['domain'],
                            "server_url": "",
                            "video_idx": 0,
                            "task_prompt": item['prompt'],
                            "protocol": item.get('protocol', ''),
                            "success": False,
                            "error": "No videos available for this item"
                        }
                        all_tasks.append(placeholder_task)
                else:
                    # 有视频，正常处理
                    for video_idx, s_url in enumerate(server_urls):
                        # 检查是否已经成功评测过
                        task_key = (item_id, video_idx)
                        if args.resume and task_key in existing_results:
                            # 跳过已成功评测的任务
                            continue
                        
                        task = prepare_evaluation_task(
                            s_url=s_url,
                            video_idx=video_idx,
                            item_data=item,
                            system_prompt=system_prompt,
                            domain_prompt=domain_prompt,
                            init_image=init_image,
                            reference_images=reference_images,
                            reference_text=reference_text,
                            fps=args.fps
                        )
                        all_tasks.append(task)
            
            logger.info(f"Prepared {len(all_tasks)} records")
            
            if args.resume:
                skipped_count = total_videos - len(all_tasks) if args.k is None else min(args.k * len(dataset), total_videos) - len(all_tasks)
                logger.info(f"Resume mode: Skipped {skipped_count} already successful evaluations")
                if len(all_tasks) == 0:
                    logger.info(f"All evaluations already completed, skipping to next file\n")
                    continue
            
            # ========== 阶段2: 多线程调用GPT评测 ==========
            results = []
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(evaluate_task_with_gpt, task, args.model_name): task 
                    for task in all_tasks
                }
                
                # 使用tqdm显示进度
                with tqdm(total=len(all_tasks), desc="Evaluating") as pbar:
                    for future in as_completed(future_to_task):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            task = future_to_task[future]
                            logger.error(f"Task failed: {e}")
                            results.append({
                                "item_id": task["item_id"],
                                "domain": task["domain"],
                                "server_url": task["server_url"],
                                "video_idx": task["video_idx"],
                                "raw_answer": "",
                                "decision": "",
                                "success": False,
                                "task_prompt": task["task_prompt"],
                                "protocol": task["protocol"],
                                "error": str(e)
                            })
                        finally:
                            pbar.update(1)
            
            # ========== 合并结果（Resume模式） ==========
            if args.resume and existing_results:
                # 合并已有的成功结果和新的评测结果
                all_results = list(existing_results.values()) + results
                logger.info(f"Resume mode: Merged {len(existing_results)} existing + {len(results)} new = {len(all_results)} total results")
            else:
                all_results = results
            
            # 保存最终结果
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            # 简单统计
            success_count = sum(1 for r in all_results if r.get('success'))
            failed_count = len(all_results) - success_count
            
            logger.info(f"  ✓ Completed: {success_count}/{len(all_results)} successful, {failed_count} failed")
            logger.info(f"  ✓ Saved to {output_filename}\n")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}\n")
            continue
    
    logger.info(f"Evaluation completed: {len(input_files)} file(s) processed")
