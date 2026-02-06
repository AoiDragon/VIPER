#!/usr/bin/env python3
"""
单文件评测结果统计脚本
从单个GPT-4o评测结果文件中计算 pass@k 准确率
"""
import json
import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def read_json(file_path: str):
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_task_name(item_id: str) -> str:
    """
    从item_id中提取task名称
    item_id格式为 {task}_{n}，例如 "object_tracking_1"
    
    Args:
        item_id: 完整的item_id
    
    Returns:
        str: task名称
    """
    parts = item_id.rsplit('_', 1)  # 从右边分割一次，避免task名称中包含下划线的情况
    return parts[0] if len(parts) > 1 else item_id


def calculate_pass_at_k(results: List[Dict], k: int = None) -> Dict:
    """
    计算 pass@k 准确率，支持decision, process_consistency, outcome_consistency三个维度
    
    Args:
        results: 评测结果列表
        k: pass@k 中的 k 值，如果为 None 则使用所有视频
    
    Returns:
        Dict: 统计结果字典
    """
    # 按 item_id 分组统计
    item_stats = defaultdict(lambda: {"videos": []})
    
    for r in results:
        if r.get('success'):
            item_id = r.get('item_id')
            video_idx = r.get('video_idx', 0)
            decision = r.get('decision')
            process_consistency = r.get('process_consistency', '')
            outcome_consistency = r.get('outcome_consistency', '')
            item_stats[item_id]["videos"].append({
                "idx": video_idx,
                "decision": decision,
                "process_consistency": process_consistency,
                "outcome_consistency": outcome_consistency
            })
    
    # 对每个 item 的视频按 idx 排序
    for item_id in item_stats:
        item_stats[item_id]["videos"].sort(key=lambda x: x["idx"])
    
    # 计算 pass@k（针对三个维度）
    use_k = k if k is not None else float('inf')
    
    # 三个维度的统计
    decision_correct = 0
    process_correct = 0
    goal_correct = 0
    total_items = 0
    
    item_details = []
    
    for item_id, stats in item_stats.items():
        total_items += 1
        
        # 只看前k个视频
        videos_to_check = stats["videos"][:use_k] if use_k != float('inf') else stats["videos"]
        
        # 判断这些视频中是否有correct（针对三个维度）
        has_decision_correct = any(v["decision"] == "correct" for v in videos_to_check)
        has_process_correct = any(v["process_consistency"] == "correct" for v in videos_to_check)
        has_goal_correct = any(v["outcome_consistency"] == "correct" for v in videos_to_check)
        
        decision_list = [v["decision"] for v in videos_to_check]
        process_list = [v["process_consistency"] for v in videos_to_check]
        goal_list = [v["outcome_consistency"] for v in videos_to_check]
        
        item_details.append({
            "item_id": item_id,
            "task": extract_task_name(item_id),
            "decision_passed": has_decision_correct,
            "process_passed": has_process_correct,
            "goal_passed": has_goal_correct,
            "decisions": decision_list,
            "process_consistency": process_list,
            "outcome_consistency": goal_list
        })
        
        if has_decision_correct:
            decision_correct += 1
        if has_process_correct:
            process_correct += 1
        if has_goal_correct:
            goal_correct += 1
    
    decision_accuracy = decision_correct / total_items if total_items > 0 else 0
    process_accuracy = process_correct / total_items if total_items > 0 else 0
    goal_accuracy = goal_correct / total_items if total_items > 0 else 0
    
    return {
        "total_items": total_items,
        "decision": {
            "correct": decision_correct,
            "incorrect": total_items - decision_correct,
            "accuracy": round(decision_accuracy, 4)
        },
        "process_consistency": {
            "correct": process_correct,
            "incorrect": total_items - process_correct,
            "accuracy": round(process_accuracy, 4)
        },
        "outcome_consistency": {
            "correct": goal_correct,
            "incorrect": total_items - goal_correct,
            "accuracy": round(goal_accuracy, 4)
        },
        "item_details": item_details
    }


def calculate_task_stats(results: List[Dict], k: int = None) -> Dict:
    """
    按task分类统计准确率，支持三个维度
    
    Args:
        results: 评测结果列表
        k: pass@k 中的 k 值
    
    Returns:
        Dict: 按task统计的结果
    """
    # 按 task 和 item_id 分组统计
    task_item_stats = defaultdict(lambda: defaultdict(lambda: {"videos": []}))
    
    for r in results:
        if r.get('success'):
            item_id = r.get('item_id')
            task_name = extract_task_name(item_id)
            video_idx = r.get('video_idx', 0)
            decision = r.get('decision')
            process_consistency = r.get('process_consistency', '')
            outcome_consistency = r.get('outcome_consistency', '')
            task_item_stats[task_name][item_id]["videos"].append({
                "idx": video_idx,
                "decision": decision,
                "process_consistency": process_consistency,
                "outcome_consistency": outcome_consistency
            })
    
    # 对每个 item 的视频按 idx 排序
    for task_name in task_item_stats:
        for item_id in task_item_stats[task_name]:
            task_item_stats[task_name][item_id]["videos"].sort(key=lambda x: x["idx"])
    
    # 计算每个task的 pass@k（针对三个维度）
    use_k = k if k is not None else float('inf')
    task_results = {}
    
    for task_name, items in task_item_stats.items():
        task_decision_correct = 0
        task_process_correct = 0
        task_goal_correct = 0
        total_items = len(items)
        
        for item_id, stats in items.items():
            # 只看前k个视频
            videos_to_check = stats["videos"][:use_k] if use_k != float('inf') else stats["videos"]
            
            # 判断这些视频中是否有correct（针对三个维度）
            has_decision_correct = any(v["decision"] == "correct" for v in videos_to_check)
            has_process_correct = any(v["process_consistency"] == "correct" for v in videos_to_check)
            has_goal_correct = any(v["outcome_consistency"] == "correct" for v in videos_to_check)
            
            if has_decision_correct:
                task_decision_correct += 1
            if has_process_correct:
                task_process_correct += 1
            if has_goal_correct:
                task_goal_correct += 1
        
        task_decision_accuracy = task_decision_correct / total_items if total_items > 0 else 0
        task_process_accuracy = task_process_correct / total_items if total_items > 0 else 0
        task_goal_accuracy = task_goal_correct / total_items if total_items > 0 else 0
        
        task_results[task_name] = {
            "total": total_items,
            "decision": {
                "correct": task_decision_correct,
                "incorrect": total_items - task_decision_correct,
                "accuracy": round(task_decision_accuracy, 4)
            },
            "process_consistency": {
                "correct": task_process_correct,
                "incorrect": total_items - task_process_correct,
                "accuracy": round(task_process_accuracy, 4)
            },
            "outcome_consistency": {
                "correct": task_goal_correct,
                "incorrect": total_items - task_goal_correct,
                "accuracy": round(task_goal_accuracy, 4)
            }
        }
    
    return task_results


def print_results(stats: Dict, task_stats: Dict, k: int, fps: float, domain: str):
    """打印统计结果，包含三个维度"""
    logger.info("")
    logger.info("="*70)
    logger.info(f"Evaluation Results - {domain}")
    logger.info("="*70)
    logger.info(f"Metric: fps@{fps}, pass@{k if k else 'all'}")
    logger.info(f"Total Items: {stats['total_items']}")
    logger.info("")
    
    # 打印三个维度的整体准确率
    logger.info("Overall Accuracy:")
    logger.info(f"  Decision:            {stats['decision']['accuracy']*100:.2f}% ({stats['decision']['correct']}/{stats['total_items']})")
    logger.info(f"  Process Consistency: {stats['process_consistency']['accuracy']*100:.2f}% ({stats['process_consistency']['correct']}/{stats['total_items']})")
    logger.info(f"  Outcome Consistency: {stats['outcome_consistency']['accuracy']*100:.2f}% ({stats['outcome_consistency']['correct']}/{stats['total_items']})")
    logger.info("")
    
    if task_stats:
        logger.info("Task-level Statistics:")
        logger.info("-"*70)
        logger.info(f"{'Task Name':<30} {'Process':<15} {'Goal':<15} {'Decision':<15}")
        logger.info("-"*70)
        for task_name, task_stat in sorted(task_stats.items()):
            process_acc = f"{task_stat['process_consistency']['accuracy']*100:.1f}%"
            goal_acc = f"{task_stat['outcome_consistency']['accuracy']*100:.1f}%"
            decision_acc = f"{task_stat['decision']['accuracy']*100:.1f}%"
            logger.info(f"{task_name:<30} {process_acc:<15} {goal_acc:<15} {decision_acc:<15}")
        logger.info("")
    

if __name__ == "__main__":
    parser = ArgumentParser(description="统计单个GPT-4o评测结果文件的准确率")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="输入JSON文件路径（GPT-4o评测结果）")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="输出JSON文件路径（默认在输入文件旁生成 _scores.json）")
    parser.add_argument("--fps", type=float, default=1.0, 
                        help="采样帧率 (用于输出文件命名和显示)")
    parser.add_argument("--k", type=int, default=None, 
                        help="pass@k (默认使用所有视频)")
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        exit(1)
    
    # 验证文件名是否匹配指定的fps和pass@k
    input_filename = os.path.basename(args.input_file)
    fps_pattern = f"fps@{args.fps}"
    pass_pattern = f"pass@{args.k}" if args.k else "pass@"
    
    if fps_pattern not in input_filename:
        logger.warning(f"Warning: Input file does not match fps@{args.fps}")
        logger.warning(f"File: {input_filename}")
        logger.warning("Proceeding anyway, but results may be inconsistent...")
    
    if args.k and pass_pattern not in input_filename:
        logger.warning(f"Warning: Input file does not match pass@{args.k}")
        logger.warning(f"File: {input_filename}")
        logger.warning("Proceeding anyway, but results may be inconsistent...")
    
    # 读取评测结果
    logger.info(f"Reading evaluation results from: {args.input_file}")
    results = read_json(args.input_file)
    
    if not results:
        logger.error("No results found in input file")
        exit(1)
    
    # 获取domain信息
    domain = results[0].get('domain', 'unknown')
    
    # 计算统计信息
    logger.info(f"Calculating statistics for {len(results)} evaluation records...")
    stats = calculate_pass_at_k(results, args.k)
    task_stats = calculate_task_stats(results, args.k)
    
    # 打印结果
    print_results(stats, task_stats, args.k, args.fps, domain)
    