<div align="center">

<h1>Beyond the Last Frame: Process-aware Evaluation for <br> Generative Video Reasoning</h1>

<div align="center">
  <a href='https://arxiv.org/abs/2512.24952'><img src='https://img.shields.io/badge/Arxiv-2512.24952-red'></a>
  <a href='https://huggingface.co/datasets/Monosail/VIPER'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-yellow'></a>
</div>

<br>

<p align="center">
  <img src="assets/outcome_hack.png" width="85%"> <br>
  <em>Illustration of outcome-hacking, where the generated video has the correct final state but an incorrect process.</em>
</p>

</div>

---

## 👀 Overview

Current video generation models often suffer from **Outcome-hacking**: they may generate a video with the correct final outcome but a wrong process. This hacks traditional single-frame evaluation metrics.

**VIPER** (VIdeo Process Evaluation for Reasoning) is designed to bridge this gap:

* **🏆 Comprehensive Benchmark:** 309 carefully curated samples spanning **6 distinct domains** (Temporal, Structural, Symbolic, Spatial, Physics, and Planning).
* **📏 New Metric (POC@r):** **P**rocess-**O**utcome **C**onsistency. We evaluate correctness at both the process and outcome levels by uniformly sampling frames at rate $r$.
* **🚫 Failure Pattern:** We identify and summarize four common failure patterns in current generative video models.

---
<div>
<p align="center">
  <img src="assets/overview.png" width="85%"> <br>
  <em>Overview of VIPER. VIPER consists of 16 tasks from 6 domains</em>
</p>
</div>

## 📊 Dataset Statistics

VIPER covers diverse reasoning tasks to ensure a holistic evaluation of video generation capabilities.

| Domain | Samples | Task Types |
| :--- | :---: | :--- |
| **Physics** | 32 | experiment, game |
| **Planning** | 44 | navigation, manipulation |
| **Spatial** | 60 | rotate, restore |
| **Structural** | 70 | chess, maze, sudoku |
| **Symbolic** | 60 | math, multimodal |
| **Temporal** | 43 | obj_move, zoom |

---

## 🚀 Quick Start

### Download
```Python
from datasets import load_dataset

# Load the full VIPER benchmark
dataset = load_dataset("Monosail/VIPER")
```

### Data Fields

- `id`: Unique identifier for the sample
- `domain`: The reasoning domain (Physics, Planning, Spatial, Structural, Symbolic, Temporal)
- `task_type`: Specific task category within the domain
- `prompt`: Text prompt describing the task
- `image`: The input image
- `reference_frames`: Ground-truth image frames
- `reference_texts`: Ground-truth text descriptions
- `protocol`: Process-level task constraints

## 🛠️ Evaluation

The evaluation pipeline is split into two stages: **inference** and **judgement**.
During **inference**, we provide scripts to generate inference outputs on the **VIPER** datasets using the following supported models:
- **Closed-source (API)**
  - Sora2  
  - Veo3.1  
  - Seedance 1.5 Pro (Opened)  
  - Wan2.6 (Opened)  
- **Open-source**
  - Wan2.2  
  - Hunyuan-1.5  
During **judgement**, we use the **OpenRouter API** and default to **gpt-5**. You may use any MLLM as long as it is compatible with the provider endpoint.

### Inference

#### Seedance 1.5 pro

To run video inference with Seedance 1.5 Pro:
```bash
bash scripts/run_sd.sh
```
Prerequisites:

- Apply for the [Seedance API](https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-seedance-1-5-pro)
- Set the environment variable `ARK_API_KEY`

#### Wan2.6

To run video inference with Wan2.6:

```bash
bash scripts/run_wan26.sh
```
Prerequisites:

- Apply for the [Wan2.6 api](https://bailian.console.aliyun.com/cn-beijing/#/home)
- Set the environment variable `DASHSCOPE_API_KEY`

### Judgement

 


## 📝 Citation

If you find our benchmark useful for your research, please consider citing:

```bibtex
@article{li2026viper,
  title={Beyond the Last Frame: Process-aware Evaluation for Generative Video Reasoning},
  author={Li, Yifan and Gu, Yukai and Min, Yingqian and Liu, Zikang and Du, Yifan and Zhou, Kun and Yang, Min and Zhao, Wayne Xin and Qiu, Minghui},
  journal={arXiv preprint arXiv:2512.24952},
  year={2025}
}
```

