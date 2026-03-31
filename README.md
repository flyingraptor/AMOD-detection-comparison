<div align="center">
   <img alt="AMOD: Arma3 Multi-view Object Detection" src="./mmrotate/Logo.svg" />
</div>

<hr>

<h3 align="center">
 Comparative Study: Oriented R-CNN vs YOLO11s-OBB on Synthetic Aerial Imagery
</h3>

<p align="center">
  <a href="#"><img alt="Python3.10+" src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch2.11" src="https://img.shields.io/badge/PyTorch-2.11-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMRotate0.3.4" src="https://img.shields.io/badge/MMRotate-0.3.4-hotpink"></a>
  <a href="#"><img alt="Ultralytics8.4" src="https://img.shields.io/badge/Ultralytics-8.4.31-purple"></a>
  <a href="#"><img alt="ARMA3" src="https://img.shields.io/badge/Dataset-AMOD_V1.0-green"></a>
</p>

<hr>

> **Fork of [unique-chan/AMOD](https://github.com/unique-chan/AMOD).**
> This repository extends the original experiment kit with a systematic two-model comparison study, full paper draft, and publication-ready figures.

---

## What is this repo?

This is a comparative study of two oriented object detection paradigms on the **AMOD** dataset — a large-scale synthetic aerial benchmark rendered in ArmA 3:

| Model | Paradigm | mAP@50 | Params | Train time |
|---|---|---|---|---|
| **YOLO11s-OBB** | Single-stage, anchor-free | **0.9040** | 9.7M | ~5h |
| Oriented R-CNN + Swin-S | Two-stage, proposal-based | 0.8952 | ~69M | ~28h |

> Full results, per-class breakdown, and analysis: [`docs/ARTICLE.md`](docs/ARTICLE.md)

---

## Repository structure

```
AMOD/
├── my_config/          # Oriented R-CNN training config (MMRotate)
├── mmrotate/           # MMRotate 0.3.4 (patched for PyTorch 2.x / RTX 5090)
├── yolo/               # YOLO11s-OBB training script + dataset YAML
├── data/               # Split index files (*.txt) — images re-downloaded separately
├── docs/
│   ├── ARTICLE.md      # Full paper draft with results and discussion
│   ├── EXPERIMENTS.md  # Experiment tracker: configs, commands, raw numbers
│   ├── plot_results.py # Generates all publication figures from training logs
│   └── figures/        # Publication-ready PNG figures (11 total)
└── images/             # Reference images used in the paper
```

---

## Dataset

AMOD V1.0 — download from HuggingFace: [`unique-chan/AMOD-V1.0`](https://huggingface.co/datasets/unique-chan/AMOD-V1.0)

Place the extracted data under `data/` so the structure matches:
```
data/
├── train/{scene_id}/{angle}/EO_{scene_id}_{angle}.png
├── test/{scene_id}/{angle}/EO_{scene_id}_{angle}.png
├── train.txt / val.txt / test.txt       ← scene ID index files (included)
└── yolo_train.txt / yolo_val.txt / ...  ← YOLO-format index files (included)
```

**12 classes used** (civilian and Boat excluded — see §3.5 of the paper):
*Armored, Artillery, Helicopter, LCU, MLRS, Plane, RADAR, SAM, Self-propelled Artillery, Support, Tank, TEL*

---

## Setup

> Tested on: Ubuntu 22.04, CUDA 13.0, RTX 5090, PyTorch 2.11.0

```bash
python -m venv venv && source venv/bin/activate

# PyTorch 2.x (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# MMDetection + MMRotate
pip install -U openmim==0.3.9
mim install mmcv-full==1.7.2
pip install -v -e mmdetection/
pip install -v -e mmrotate/

# YOLO
pip install ultralytics==8.4.31

# Utilities
pip install tensorboard matplotlib pandas
```

> **Note:** The mmrotate source in this repo contains patches for PyTorch 2.x API compatibility, RTX 5090 (sm_120) numerical stability, and empty-annotation edge cases. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full list of changes.

---

## Training

### Oriented R-CNN + Swin-S

```bash
python mmrotate/tools/train.py \
  my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
  --work-dir work_dirs/orientedrcnn_swinS_baseline
```

### YOLO11s-OBB

```bash
python yolo/train_yolo26_obb.py \
  --model yolo11s-obb \
  --data yolo/amod_yolo.yaml \
  --epochs 30 --batch 4 --imgsz 1024 \
  --project runs/yolo_obb --name yolo11s_baseline
```

---

## Evaluation

### Oriented R-CNN (full val set)

```bash
python mmrotate/tools/test.py \
  my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
  work_dirs/orientedrcnn_swinS_baseline/best_mAP_epoch_30.pth \
  --eval mAP \
  --cfg-options data.test.ann_file=val.txt data.test.img_prefix=train
```

### YOLO11s-OBB (full val set)

```bash
python -c "
from ultralytics import YOLO
m = YOLO('runs/yolo_obb/yolo11s_baseline/weights/best.pt')
m.val(data='yolo/amod_yolo.yaml', imgsz=1024, batch=4, split='val')
"
```

---

## Results

See [`docs/ARTICLE.md §5`](docs/ARTICLE.md) for the full comparison table and per-class AP breakdown.
See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for raw numbers, training logs, and all commands used.

Figures are in [`docs/figures/`](docs/figures/) and can be regenerated with:

```bash
python docs/plot_results.py
```

---

## Citation

If you use the AMOD dataset, please cite the original repository:

```
unique-chan. (2024). AMOD: Arma3 Multi-view Object Detection — Experiment Kit.
GitHub repository. https://github.com/unique-chan/AMOD
```
