#!/usr/bin/env python3
"""
评测结果统计脚本
从GPT-4o评测结果中按domain和task计算 pass@k 准确率
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


def calculate_pass_at_k(results: List[Dict], k: int = None) -> Dict:
    """
    计算 pass@k 准确率，支持decision, process_consistency, outcome_consistency三个维度
    
    Args:
        results: 评测结果列表
        k: pass@k 中的 k 值，如果为 None 则使用所有视频
    
    Returns:
        Dict: 统计结果字典
    """
    # 按 task 分组统计
    task_stats = defaultdict(lambda: {"videos": []})
    all_task_ids = set()  # 记录所有出现过的task_id
    
    for r in results:
        task_id = r.get('item_id')
        all_task_ids.add(task_id)  # 无论success与否，都记录task_id
        
        if r.get('success'):
            video_idx = r.get('video_idx', 0)
            decision = r.get('decision')
            process_consistency = r.get('process_consistency', '')
            outcome_consistency = r.get('outcome_consistency', '')
            task_stats[task_id]["videos"].append({
                "idx": video_idx,
                "decision": decision,
                "process_consistency": process_consistency,
                "outcome_consistency": outcome_consistency
            })
    
    # 对每个 task 的视频按 idx 排序
    for task_id in task_stats:
        task_stats[task_id]["videos"].sort(key=lambda x: x["idx"])
    
    # 计算 pass@k（针对三个维度）
    use_k = k if k is not None else float('inf')
    
    decision_correct = 0
    process_correct = 0
    goal_correct = 0
    
    # 遍历所有task_id（包括没有成功视频的）
    for task_id in all_task_ids:
        if task_id in task_stats:
            # 只看前k个视频
            videos_to_check = task_stats[task_id]["videos"][:use_k] if use_k != float('inf') else task_stats[task_id]["videos"]
            
            # 判断这些视频中是否有correct（针对三个维度）
            has_decision_correct = any(v["decision"] == "correct" for v in videos_to_check)
            has_process_correct = any(v["process_consistency"] == "correct" for v in videos_to_check)
            has_goal_correct = any(v["outcome_consistency"] == "correct" for v in videos_to_check)
            
            if has_decision_correct:
                decision_correct += 1
            if has_process_correct:
                process_correct += 1
            if has_goal_correct:
                goal_correct += 1
        # else: 没有成功视频的task，自动算作0个correct
    
    total_tasks = len(all_task_ids)  # 使用所有task_id的数量作为分母
    decision_accuracy = decision_correct / total_tasks if total_tasks > 0 else 0
    process_accuracy = process_correct / total_tasks if total_tasks > 0 else 0
    goal_accuracy = goal_correct / total_tasks if total_tasks > 0 else 0
    
    return {
        "total_tasks": total_tasks,
        "decision": {
            "passed_tasks": decision_correct,
            "accuracy": round(decision_accuracy, 4)
        },
        "process_consistency": {
            "passed_tasks": process_correct,
            "accuracy": round(process_accuracy, 4)
        },
        "outcome_consistency": {
            "passed_tasks": goal_correct,
            "accuracy": round(goal_accuracy, 4)
        }
    }


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
    all_task_items = defaultdict(set)  # 记录每个task下所有的item_id
    
    for r in results:
        item_id = r.get('item_id')
        task_name = extract_task_name(item_id)
        all_task_items[task_name].add(item_id)  # 无论success与否，都记录item_id
        
        if r.get('success'):
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
    
    for task_name in all_task_items:
        task_decision_correct = 0
        task_process_correct = 0
        task_goal_correct = 0
        total_items = len(all_task_items[task_name])  # 使用所有item_id的数量作为分母
        
        for item_id in all_task_items[task_name]:
            if item_id in task_item_stats[task_name]:
                # 只看前k个视频
                videos_to_check = task_item_stats[task_name][item_id]["videos"][:use_k] if use_k != float('inf') else task_item_stats[task_name][item_id]["videos"]
                
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
            # else: 没有成功视频的item，自动算作0个correct
        
        task_decision_accuracy = task_decision_correct / total_items if total_items > 0 else 0
        task_process_accuracy = task_process_correct / total_items if total_items > 0 else 0
        task_goal_accuracy = task_goal_correct / total_items if total_items > 0 else 0
        
        task_results[task_name] = {
            "total": total_items,
            "decision": {
                "passed": task_decision_correct,
                "accuracy": round(task_decision_accuracy, 4)
            },
            "process_consistency": {
                "passed": task_process_correct,
                "accuracy": round(task_process_accuracy, 4)
            },
            "outcome_consistency": {
                "passed": task_goal_correct,
                "accuracy": round(task_goal_accuracy, 4)
            }
        }
    
    return task_results


def process_file(input_file: str, k: int = None, fps: float = None) -> Dict:
    """
    处理单个评测结果文件（可能包含一个或多个task）
    
    Args:
        input_file: 输入文件路径
        k: pass@k 中的 k 值
        fps: fps@n 中的 n 值
    
    Returns:
        Dict: 统计结果，包含domain名称和各task统计（不包含average）
    """
    
    # 读取评测结果
    results = read_json(input_file)
    domain = results[0]["domain"]

    # 计算 pass@k (domain级别)
    pass_at_k_stats = calculate_pass_at_k(results, k)
    
    # 计算按task分类的统计
    task_stats = calculate_task_stats(results, k)
    
    decision_acc = pass_at_k_stats['decision']['accuracy']
    decision_passed = pass_at_k_stats['decision']['passed_tasks']
    total = pass_at_k_stats['total_tasks']
    
    logger.info(f"{domain} - {os.path.basename(input_file)}: Decision={decision_acc*100:.1f}% ({decision_passed}/{total})")
    
    return {
        "domain": domain,
        "task_stats": task_stats  # 不在这里计算average，留到合并后统一计算
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="计算评测结果的 pass@k 准确率（按domain统计）")
    parser.add_argument("--input_path", type=str, required=True, 
                        help="输入路径，可以是单个JSON文件或包含多个JSON文件的目录")
    parser.add_argument("--output_path", type=str, default=None, 
                        help="输出目录路径（默认与输入文件同目录）")
    parser.add_argument("--fps", type=float, default=1.0, 
                        help="采样帧率")
    parser.add_argument("--k", type=int, default=8, 
                        help="pass@k")
    args = parser.parse_args()
    
    # 确定输入文件列表
    input_files = []
    if os.path.isfile(args.input_path):
        input_files = [args.input_path]
        default_output_dir = os.path.dirname(args.input_path)
    elif os.path.isdir(args.input_path):
        # 构建文件名匹配模式：必须包含 fps@{fps} 和 pass@{k}
        fps_pattern = f"fps@{args.fps}"
        pass_pattern = f"pass@{args.k}"
        
        input_files = [
            os.path.join(args.input_path, f)
            for f in sorted(os.listdir(args.input_path))
            if f.endswith('.json') 
            and not f.endswith('_scores.json')  # 排除已有的统计文件
            and fps_pattern in f  # 必须包含指定的fps
            and pass_pattern in f  # 必须包含指定的pass@k
        ]
        default_output_dir = args.input_path
    else:
        logger.error(f"Invalid input_path: {args.input_path}")
        exit(1)
    
    if not input_files:
        logger.error(f"No JSON files found in {args.input_path}")
        exit(1)
    
    # 确定输出目录
    output_dir = args.output_path if args.output_path else default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定指标名称
    metric_name = f"fps@{args.fps}_pass@{args.k}"
    
    logger.info(f"{'='*60}")
    logger.info(f"Video Reasoning Bench - Scoring")
    logger.info(f"{'='*60}")
    logger.info(f"Metric: {metric_name}")
    logger.info(f"Total files: {len(input_files)}")
    
    # 处理每个文件（每个文件可能代表一个domain中的一个或多个task）
    domain_results = defaultdict(dict)  # {domain: {task: stats}}
    
    for input_file in input_files:
        try:
            result = process_file(input_file, args.k, args.fps)
            domain = result['domain']
            
            # 合并task统计到对应的domain下
            for task_name, task_stat in result['task_stats'].items():
                if task_name in domain_results[domain]:
                    # 如果task已存在，说明有重复，这里选择覆盖（或者可以报警告）
                    logger.warning(f"Task '{task_name}' already exists in domain '{domain}', overwriting...")
                domain_results[domain][task_name] = task_stat
                
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            continue
    
    # 将defaultdict转换为普通dict
    domain_results = dict(domain_results)
    
    # 为每个domain计算average（三个维度）
    for domain in domain_results:
        task_stats = domain_results[domain]
        if task_stats:
            # 分别计算三个维度的平均准确率
            decision_accs = [t["decision"]["accuracy"] for t in task_stats.values()]
            process_accs = [t["process_consistency"]["accuracy"] for t in task_stats.values()]
            goal_accs = [t["outcome_consistency"]["accuracy"] for t in task_stats.values()]
            
            task_stats['average'] = {
                "total": sum(t["total"] for t in task_stats.values()),
                "decision": {
                    "passed": sum(t["decision"]["passed"] for t in task_stats.values()),
                    "accuracy": round(sum(decision_accs) / len(decision_accs), 4)
                },
                "process_consistency": {
                    "passed": sum(t["process_consistency"]["passed"] for t in task_stats.values()),
                    "accuracy": round(sum(process_accs) / len(process_accs), 4)
                },
                "outcome_consistency": {
                    "passed": sum(t["outcome_consistency"]["passed"] for t in task_stats.values()),
                    "accuracy": round(sum(goal_accs) / len(goal_accs), 4)
                }
            }
            domain_results[domain] = task_stats
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Summary by Domain and Task")
    logger.info(f"{'='*80}\n")
    
    # 打印每个domain的详细统计
    for domain in sorted(domain_results.keys()):
        task_stats = domain_results[domain]
        if 'average' in task_stats:
            avg_stats = task_stats['average']
            logger.info(f"{domain}:")
            logger.info(f"  {'Task':<30} {'Process':<15} {'Outcome':<15} {'Decision':<15}")
            logger.info(f"  {'-'*75}")
            
            for task_name in sorted(task_stats.keys()):
                if task_name != 'average':
                    stats = task_stats[task_name]
                    decision_acc = f"{stats['decision']['accuracy']*100:.1f}%"
                    process_acc = f"{stats['process_consistency']['accuracy']*100:.1f}%"
                    outcome_acc = f"{stats['outcome_consistency']['accuracy']*100:.1f}%"
                    logger.info(f"  {task_name:<30} {process_acc:<15} {outcome_acc:<15} {decision_acc:<15}")
            
            # 打印平均值
            decision_avg = f"{avg_stats['decision']['accuracy']*100:.1f}%"
            process_avg = f"{avg_stats['process_consistency']['accuracy']*100:.1f}%"
            outcome_avg = f"{avg_stats['outcome_consistency']['accuracy']*100:.1f}%"
            logger.info(f"  {'-'*75}")
            logger.info(f"  {'average':<30} {process_avg:<15} {outcome_avg:<15} {decision_avg:<15}")
            logger.info("")
    
    logger.info(f"{'='*80}\n")
        
    #     # 计算总平均准确率
    #     if domain_avg_accuracies:
    #         overall_avg = sum(domain_avg_accuracies) / len(domain_avg_accuracies)
    #         total_passed = sum(task_stats['average']['passed'] for task_stats in domain_results.values() if 'average' in task_stats)
    #         total_tasks = sum(task_stats['average']['total'] for task_stats in domain_results.values() if 'average' in task_stats)
            
    #         logger.info(f"{'='*60}")
    #         logger.info(f"Overall Average: {overall_avg*100:.1f}% ({total_passed}/{total_tasks})")
    #         logger.info(f"{'='*60}\n")
            
    #         # 打印每个domain的详细统计
    #         logger.info(f"Domain-level Statistics:")
    #         logger.info(f"{'='*60}")
    #         for domain in sorted(domain_results.keys()):
    #             task_stats = domain_results[domain]
    #             if 'average' in task_stats:
    #                 avg_stats = task_stats['average']
    #                 logger.info(f"\n{domain}:")
    #                 for task_name in sorted(task_stats.keys()):
    #                     if task_name != 'average':
    #                         stats = task_stats[task_name]
    #                         logger.info(f"  {task_name}: {stats['accuracy']*100:.1f}% ({stats['passed']}/{stats['total']})")
    #                 logger.info(f"  average: {avg_stats['accuracy']*100:.1f}% ({avg_stats['passed']}/{avg_stats['total']})")
    #         logger.info(f"{'='*60}\n")
        
    #     # 保存结果，文件名不包含时间戳，使用固定的格式
    #     output_file = os.path.join(output_dir, f"{metric_name}_scores.json")
        
    #     # 如果文件已存在，先读取已有结果并合并
    #     if os.path.exists(output_file):
    #         try:
    #             with open(output_file, 'r', encoding='utf-8') as f:
    #                 existing_results = json.load(f)
                
    #             logger.info(f"Merging with existing results from {output_file}")
                
    #             # 智能合并：对于每个domain，合并task级别的数据
    #             for domain, new_task_stats in domain_results.items():
    #                 if domain in existing_results:
    #                     # Domain已存在，合并task
    #                     existing_tasks = existing_results[domain]
    #                     # 移除旧的average（稍后重新计算）
    #                     if 'average' in existing_tasks:
    #                         del existing_tasks['average']
    #                     if 'average' in new_task_stats:
    #                         del new_task_stats['average']
                        
    #                     # 合并task统计
    #                     existing_tasks.update(new_task_stats)
                        
    #                     # 重新计算average
    #                     if existing_tasks:
    #                         task_avg_accuracy = sum(t["accuracy"] for t in existing_tasks.values()) / len(existing_tasks)
    #                         existing_tasks['average'] = {
    #                             "total": sum(t["total"] for t in existing_tasks.values()),
    #                             "passed": sum(t["passed"] for t in existing_tasks.values()),
    #                             "accuracy": round(task_avg_accuracy, 4)
    #                         }
                        
    #                     existing_results[domain] = existing_tasks
    #                 else:
    #                     # Domain不存在，直接添加
    #                     existing_results[domain] = new_task_stats
                
    #             domain_results = existing_results
                
    #         except Exception as e:
    #             logger.warning(f"Failed to read/merge existing results: {e}, will overwrite")
        
    #     # 保存合并后的结果
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         json.dump(domain_results, f, ensure_ascii=False, indent=2)
        
    #     logger.info(f"Results saved to: {output_file}")
    # else:
    #     logger.error("No valid results to save")
    #     exit(1)

