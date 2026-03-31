# AMOD Experiment Tracker

## Project Goal
Compare **OrientedRCNN + SwinTransformer-S** (two-stage) vs **YOLO11s-OBB** (single-stage)
on the AMOD dataset (synthetic aerial imagery with oriented bounding boxes).

Produce a full comparison across: accuracy, speed, recall, and training cost.

---

## Environment

```bash
cd ~/Repositories/AMOD
source venv/bin/activate          # PyTorch 2.11.0+cu130, CUDA 13.0, RTX 5090 (sm_120)
```

| Package      | Version  |
|---|---|
| PyTorch      | 2.11.0+cu130 |
| ultralytics  | 8.4.31 (YOLO11s-OBB used; no YOLO26 release existed at time of experiment) |
| mmcv-full    | 1.7.2 (built from source at `/tmp/mmcv-1.7.2`) |
| mmdetection  | 2.28.2 |
| mmrotate     | 0.3.4 |

---

## Dataset — AMOD V1.0

- **Source**: HuggingFace `unique-chan/AMOD-V1.0`
- **Location**: `data/train/`, `data/test/`
- **Structure**: `data/{split}/{scene_id}/{angle}/EO_{scene_id}_{angle}.png` + CSV annotations
- **Size**: ~24,245 training images, ~6,000 test images
- **Classes**: 12 (vehicles, aircraft, etc.)
- **Annotation format**: Oriented bounding boxes (polygon → OBB)
- **Angles sampled**: 0°, 10°, 20°, 30°, 40°, 50° per scene

### YOLO label conversion
```bash
python yolo/convert_to_yolo_obb.py    # generates data/{split}/{scene}/{angle}/*.txt
```

---

## Training Commands

### OrientedRCNN + Swin-S (Baseline)
```bash
python mmrotate/tools/train.py \
  "my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py" \
  --work-dir work_dirs/orientedrcnn_swinS_baseline
```

### YOLO11s-OBB (Baseline)
```bash
python yolo/train_yolo26_obb.py \
  --model yolo11s-obb \
  --epochs 30 \
  --batch 4 \
  --imgsz 1024
```

### TensorBoard (monitor both)
```bash
tensorboard --logdir work_dirs/    # shows all runs under work_dirs/
```

---

## Baseline Configurations

### OrientedRCNN — `my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py`

| Parameter          | Value |
|---|---|
| Backbone           | SwinTransformer-S (pretrained ImageNet-1K) |
| Neck               | FPN (4 levels) |
| Head               | OrientedRPNHead + RotatedShared2FCBBoxHead |
| Epochs             | 30 |
| Batch size         | 4 (samples_per_gpu=4) |
| Optimizer          | SGD lr=0.005, momentum=0.9, wd=0.0001 |
| LR schedule        | Step decay (epochs 16, 22) with linear warmup 500 iter |
| AMP (fp16)         | Yes (Fp16OptimizerHook, dynamic loss scale) |
| Image size         | 1024×1024 crop (input: 1920×1440, scale range 0.8–1.2×) |
| Angle convention   | le90 |
| Workers            | 4 per GPU |
| Eval interval      | Every 5 epochs (on val_mini.txt — 1,020 images, 170 scenes) |
| Checkpoint         | Every 5 epochs + best mAP (saved to `work_dirs/orientedrcnn_swinS_baseline/`) |
| Max detections     | 200 per image (test_cfg, for faster pycocotools eval) |
| Val set (training) | `data/val_mini.txt` — 170 randomly sampled scenes (seed=42), ~1,020 images |
| Val set (final)    | `data/val.txt` — full 1,041 scenes, ~6,246 images (run manually after training) |

### YOLO11s-OBB — `yolo/train_yolo26_obb.py`

| Parameter          | Value |
|---|---|
| Architecture       | YOLO11s-OBB (single-stage, anchor-free, NMS-free) |
| Backbone           | YOLO11s pretrained on COCO |
| Parameters         | 9.7M (9,703,431) |
| GFLOPs             | 22.3 |
| Epochs             | 30 |
| Batch size         | 4 |
| Optimizer          | SGD lr=0.005, momentum=0.9, wd=0.0001 |
| LR schedule        | Cosine annealing (lr0=0.005 → lrf=lr0×0.01=5e-5) |
| AMP (fp16)         | Yes (auto, ultralytics default) |
| Image size         | 1024×1024 |
| Warmup epochs      | 3.0 |
| Rotation aug       | ±180° (full aerial rotation) |
| Flip aug           | ud=0.5, lr=0.5 |
| Val set (training) | `data/yolo_val_mini.txt` — same 170 scenes as RCNN mini-val (~1,020 images) |
| Checkpoint         | Every 5 epochs + best/last always saved |

---

## Metrics Collected

| Metric | Oriented R-CNN | YOLO11s-OBB | Notes |
|---|---|---|---|
| `mAP@50` | ✅ 0.8952 | ✅ 0.9040 | Primary metric |
| `mAP@50:95` | — | ✅ 0.671 | MMRotate only reports mAP@50 |
| Per-class AP@50 | ✅ (12 classes) | ✅ (12 classes) | See full val tables below |
| Precision | — | ✅ 0.889 | Not reported by MMRotate |
| Recall | — | ✅ 0.834 | Not reported by MMRotate |
| FPS | — | ✅ ~256 (3.9ms) | RCNN not benchmarked |
| Training time | ✅ 28h | ✅ 5h | Wall-clock, RTX 5090 (confirmed) |
| Parameters (M) | ✅ ~69M | ✅ 9.7M | Architecture-level |
| GFLOPs | ✅ ~190 | ✅ 22.3 | At model-summary resolution |

---

## Evaluation Commands

### OrientedRCNN — evaluate on full val set ✅ (used for paper)
```bash
python mmrotate/tools/test.py \
  "my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py" \
  work_dirs/orientedrcnn_swinS_baseline/best_mAP_epoch_30.pth \
  --eval mAP \
  --cfg-options data.test.ann_file=val.txt data.test.img_prefix=train
```

> ⚠️ Must override both `ann_file` and `img_prefix` — validation images live under `data/train/`, not `data/test/`.
> Output saved to: `work_dirs/orientedrcnn_swinS_baseline/full_val_results.log`

### YOLO11s — evaluate on full val set ✅ (used for paper)
```python
from ultralytics import YOLO
model = YOLO("runs/yolo_obb/yolo11s_baseline/weights/best.pt")
metrics = model.val(data="yolo/amod_yolo.yaml", split="val", imgsz=1024, batch=4)
print(metrics)
```

> ⚠️ The AMOD test split has **no ground truth annotations** (labels held out, competition style).
> `val.txt` / `yolo_val.txt` (6,240 images) is the definitive evaluation set for both models.

### YOLO26s-OBB — `yolo/train_yolo26_obb.py`

| Parameter          | Value |
|---|---|
| Architecture       | YOLO26s-OBB (single-stage, NMS-free, dual-head OBB26) |
| Backbone           | YOLO26s C3k2+SPPF+C2PSA pretrained on COCO |
| Head               | `OBB26` — dedicated OBB head with specialized angle loss |
| Parameters         | 10,529,510 (~10.5M) — model summary pre-fuse |
| GFLOPs             | 24.5 — model summary at 1024px |
| Ultralytics        | 8.4.33 |
| Epochs             | 30 |
| Batch size         | 4 |
| Optimizer          | SGD lr=0.005, momentum=0.9, wd=0.0001 |
| LR schedule        | Cosine annealing (lr0=0.005 → lrf=lr0×0.01=5e-5) |
| AMP (fp16)         | Yes (auto) |
| Image size         | 1024×1024 |
| Warmup epochs      | 3.0 |
| Rotation aug       | ±180° |
| Flip aug           | ud=0.5, lr=0.5 |
| Loss components    | box_loss, cls_loss, dfl_loss, **angle_loss** (new vs YOLO11s) |
| Val set (training) | `data/yolo_val_mini.txt` — same 170 scenes as RCNN/YOLO11s mini-val |
| Checkpoint         | Every 5 epochs + best/last always saved |

> Training launched: ~epoch 1 in progress. ~5h expected (same as YOLO11s).

---

## Results Table

### Baseline (30 epochs, batch=4, imgsz=1024) ✅ FINAL

| Model | mAP@50 | mAP@50:95 | Precision | Recall | FPS | Train (h) | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|---|---|
| OrientedRCNN + Swin-S | 0.8952 | — ¹ | — ¹ | — ¹ | — ¹ | ~28 | ~69 | ~190 |
| YOLO11s-OBB | 0.9040 | 0.671 | 0.889 | 0.834 | ~256 (3.9ms) | ~5 | 9.7 | 22.3 |
| **YOLO26s-OBB** | **🔄 training** | — | — | — | — | ~5 | **10.5** | **24.5** |

¹ MMRotate evaluation reports mAP@50 only; FPS not benchmarked for RCNN.

> ✅ YOLO11s-OBB and Oriented R-CNN fully evaluated.
> 🔄 YOLO26s-OBB training in progress — expected ~5h, same hyperparameters for fair comparison.

---

## Intermediate Results

### OrientedRCNN — Epoch 10 / 30 — first successful eval (val_mini.txt, 1,020 images, IoU=0.5)

> **Note:** Epoch 5 evaluation was attempted but crashed with `IndexError: too many indices for array` in `eval_map.py` — caused by empty bbox arrays having shape `(0,)` instead of `(0, 5)`. Fixed in `mmrotate/datasets/amod.py` (`.reshape(-1, 5)`) and `eval_map.py` (ndim guard). Training resumed from `epoch_5.pth`. Epoch 10 is the **first successful** mAP result.

| Class | GTs | Dets | Recall | AP |
|---|---|---|---|---|
| Armored | 1636 | 2040 | 0.9285 | 0.8951 |
| Artillery | 66 | 106 | 0.9848 | 0.9077 |
| Helicopter | 486 | 579 | 0.8745 | **0.7877** ← lowest |
| LCU | 6 | 23 | 1.0000 | 1.0000 |
| MLRS | 126 | 163 | 0.9683 | 0.9036 |
| Plane | 500 | 590 | 0.9780 | 0.9091 |
| RADAR | 234 | 397 | 0.9231 | 0.8858 |
| SAM | 216 | 338 | 0.8843 | 0.8166 |
| Self-propelled Artillery | 60 | 353 | 0.9500 | 0.9028 |
| Support | 929 | 1098 | 0.9462 | 0.9032 |
| Tank | 649 | 734 | 0.9353 | 0.9048 |
| TEL | 96 | 108 | 0.9375 | 0.8950 |
| **mAP@50** | | | | **0.8926** |

**Notes:**
- Evaluated on `val_mini.txt` (170 scenes, 1,020 images) — not final numbers
- LCU AP=1.0 is unreliable (only 6 GT instances in mini-val)
- Helicopter lowest AP — complex shape from above, rotor occlusion
- SAM: 338 dets for 216 GTs — false positives present
- Self-propelled Artillery: 353 dets for 60 GTs — high FP rate, but AP still 0.90
- 20 epochs remain; LR drops at epochs 16 and 22 will improve precision

### OrientedRCNN — Epoch 15 / 30 (val_mini.txt, 1,020 images, IoU=0.5)

> LR still at 5e-3. Step decay fires at epoch 16.

| Class | GTs | Dets | Recall | AP | Δ vs Ep10 |
|---|---|---|---|---|---|
| Armored | 1636 | 2141 | 0.9291 | 0.8987 | +0.004 |
| Artillery | 66 | 84 | 1.0000 | 0.9986 | **+0.091 ↑↑** |
| Helicopter | 486 | 532 | 0.9095 | 0.8767 | **+0.089 ↑↑** |
| LCU | 6 | 13 | 1.0000 | 1.0000 | — |
| MLRS | 126 | 172 | 0.9603 | 0.9052 | +0.002 |
| Plane | 500 | 516 | 0.9880 | 0.9089 | ±0 |
| RADAR | 234 | **872** | 0.9188 | 0.8648 | -0.021 ↓ ⚠️ high FP |
| SAM | 216 | **467** | 0.8657 | 0.8038 | -0.013 ↓ ⚠️ high FP |
| Self-propelled Artillery | 60 | 124 | 0.9000 | 0.8819 | -0.021 ↓ |
| Support | 929 | 1209 | 0.9440 | 0.8959 | -0.007 |
| Tank | 649 | 721 | 0.9353 | 0.9058 | +0.001 |
| TEL | 96 | 100 | 0.9479 | 0.9039 | +0.009 |
| **mAP@50** | | | | **0.9037** | **+0.011** |

**Notes:**
- Big gains in Artillery (+9.1%) and Helicopter (+8.9%) — hardest classes improving well
- RADAR: 872 dets for 234 GTs (3.7× ratio) — serious false positive problem
- SAM: 467 dets for 216 GTs (2.2× ratio) — also high FP
- LR drop at epoch 16 expected to reduce false positives by forcing more conservative predictions
- `best_mAP.pth` updated (0.9037 > 0.8926)

### OrientedRCNN — Epoch 30 / 30 — FINAL (val_mini.txt, 1,020 images, IoU=0.5)

> ⚠️ These are mini-val numbers. Run full eval on `val.txt` for paper-quality results.

| Class | GTs | Dets | Recall | AP | Δ vs Ep15 |
|---|---|---|---|---|---|
| Armored | 1636 | 1790 | 0.9315 | 0.9006 | +0.002 |
| Artillery | 66 | 98 | 1.0000 | **1.0000** | +0.001 |
| Helicopter | 486 | 534 | 0.9218 | 0.8829 | +0.006 |
| LCU | 6 | 8 | 1.0000 | 1.0000 | — |
| MLRS | 126 | 152 | 0.9603 | 0.9067 | +0.002 |
| Plane | 500 | 515 | 0.9880 | 0.9087 | ±0 |
| RADAR | 234 | 362 ← was 872 | 0.9231 | 0.8935 | **+0.029 ↑** |
| SAM | 216 | 266 ← was 467 | 0.8889 | 0.8140 | +0.010 |
| Self-propelled Artillery | 60 | 90 ← was 124 | 0.9333 | 0.9043 | +0.022 |
| Support | 929 | 1004 | 0.9548 | 0.9068 | +0.011 |
| Tank | 649 | 718 | 0.9430 | 0.9061 | +0.001 |
| TEL | 96 | 99 | 0.9688 | 0.9081 | +0.004 |
| **mAP@50** | | | | **0.9110** | **+0.007** |

**Key observations:**
- LR decay at epochs 16 & 22 dramatically reduced false positives (RADAR FP: 3.7×→1.5×, SAM: 2.2×→1.2×)
- SAM remains the hardest class (AP 0.814) — visually similar to other vehicle types
- Helicopter improved consistently across all epochs (0.788→0.877→0.883)
- Best checkpoint: `best_mAP_epoch_25.pth` or `epoch_30.pth`

**Next step: run full evaluation on `val.txt` (6,246 images) for paper numbers**

---

### OrientedRCNN — Full Val (val.txt, 6,246 images) — PAPER NUMBERS ✅

| Class | GTs | Dets | Recall | AP@50 |
|---|---|---|---|---|
| Armored | 9422 | 10648 | 0.9311 | 0.9005 |
| Artillery | 245 | 334 | 0.9592 | 0.9036 |
| Helicopter | 2914 | 3134 | 0.8844 | **0.7946** ← lowest |
| LCU | 120 | 170 | 0.9917 | 0.9091 |
| MLRS | 898 | 1075 | 0.9543 | 0.9076 |
| Plane | 2820 | 2941 | 0.9780 | 0.9085 |
| RADAR | 1798 | 2528 | 0.9255 | 0.8957 |
| SAM | 1660 | 2023 | 0.9530 | 0.9025 |
| Self-propelled Artillery | 517 | 727 | 0.9555 | 0.9068 |
| Support | 5659 | 6180 | 0.9458 | 0.9055 |
| Tank | 5207 | 5713 | 0.9389 | 0.9053 |
| TEL | 512 | 588 | 0.9688 | 0.9034 |
| **mAP@50** | | | | **0.8952** |

**Key observations:**
- Helicopter is the hardest class (AP=0.7946) — complex aerial silhouette
- All other 11 classes cluster tightly between 0.895–0.909 — extremely consistent
- Full val mAP (0.8952) is lower than mini-val (0.9110) — mini-val sample was slightly easier (expected)

---

### YOLO11s-OBB — Full Val (val.txt, 6,240 images) — PAPER NUMBERS ✅

| Class | Images | Instances | Precision | Recall | AP@50 | AP@50:95 |
|---|---|---|---|---|---|---|
| Armored | 3135 | 9309 | 0.828 | 0.877 | 0.890 | 0.625 |
| Artillery | 156 | 245 | 0.948 | 0.824 | 0.906 | 0.623 |
| Helicopter | 1415 | 2896 | 0.960 | 0.921 | **0.967** | 0.743 |
| LCU | 66 | 120 | 0.935 | 0.975 | 0.983 | 0.833 |
| MLRS | 396 | 884 | 0.769 | 0.689 | 0.804 | 0.616 |
| Plane | 1294 | 2800 | 0.992 | 0.977 | **0.994** | 0.879 |
| RADAR | 872 | 1783 | 0.891 | 0.865 | 0.927 | 0.619 |
| SAM | 764 | 1641 | 0.926 | 0.874 | 0.937 | 0.678 |
| Self-propelled Artillery | 324 | 500 | 0.658 | 0.634 | **0.690** ← lowest | 0.474 |
| Support | 2306 | 5548 | 0.920 | 0.598 | 0.850 | 0.615 |
| Tank | 2071 | 5165 | 0.954 | 0.891 | 0.954 | 0.698 |
| TEL | 260 | 512 | 0.889 | 0.883 | 0.943 | 0.654 |
| **all** | **6240** | **31403** | **0.889** | **0.834** | **0.904** | **0.671** |

**Speed:** 3.9ms inference per image = **~256 FPS** (batch=8, RTX 5090)

---

### YOLO11s-OBB — Epoch 30 / 30 — mini-val (val_mini.txt, 1,020 images)

> ⚠️ Mini-val numbers. Run full eval on `val.txt` for paper-quality results.

| Class | Images | Instances | Precision | Recall | AP@50 | AP@50:95 |
|---|---|---|---|---|---|---|
| Armored | 544 | 1605 | 0.794 | 0.867 | 0.878 | 0.638 |
| Artillery | 36 | 66 | 0.984 | 0.952 | **0.980** | 0.719 |
| Helicopter | 237 | 486 | 0.966 | 0.942 | 0.973 | 0.775 |
| LCU | 6 | 6 | 0.938 | 1.000 | **0.995** | 0.739 |
| MLRS | 42 | 118 | 0.754 | 0.839 | 0.919 | 0.707 |
| Plane | 213 | 496 | 0.989 | 0.994 | **0.995** | 0.880 |
| RADAR | 102 | 233 | 0.897 | 0.854 | 0.932 | 0.596 |
| SAM | 108 | 209 | 0.911 | 0.832 | 0.918 | 0.640 |
| Self-propelled Artillery | 58 | 58 | 0.516 | 0.517 | **0.480** ← lowest | 0.370 |
| Support | 410 | 916 | 0.916 | 0.537 | 0.827 | 0.606 |
| Tank | 291 | 640 | 0.947 | 0.902 | 0.954 | 0.706 |
| TEL | 42 | 96 | 0.904 | 0.886 | 0.962 | 0.711 |
| **all** | **1020** | **4929** | **0.876** | **0.844** | **0.901** | **0.674** |

**Speed:** 0.2ms preprocess + 2.4ms inference + 1.1ms postprocess = **3.7ms/image ≈ 270 FPS** (batch=1)

**Key observations:**
- Self-propelled Artillery (AP=0.480) is by far the hardest class — same trend as RCNN (AP=0.904 for RCNN vs 0.480 for YOLO — large gap, likely due to small/irregular appearance)
- Support recall=0.537 is low — class has large visual diversity (various support vehicles)
- Plane, LCU, Artillery near-perfect (0.995, 0.995, 0.980)
- YOLO trades slightly lower AP on hard classes for dramatically faster inference

---

## Scope

This study focuses on the **two-baseline comparison** (Oriented R-CNN vs YOLO11s-OBB). Ablation experiments and per-angle breakdowns were considered but dropped to keep the paper focused. They remain as future work.

---

## Current Status

| Run | Status | Notes |
|---|---|---|
| OrientedRCNN-B | ✅ **COMPLETE** — mAP@50: **0.8952** (full val, 6,246 img) | **28h** wall-clock, fp16, batch=4. |
| YOLO11s-B | ✅ **COMPLETE** — mAP@50: **0.9040** (full val, 6,240 img) | **5h** wall-clock, fp16, batch=4. |
| YOLO26s-B | 🔄 **TRAINING** — epoch 1/30 | Ultralytics 8.4.33, same hyps as YOLO11s. Record exact time from final log line. |

---

## Key Files Changed from Original Repo

| File | Change | Reason |
|---|---|---|
| `mmrotate/mmrotate/datasets/amod.py` | Path construction, bbox/label filtering | Match V1.0 dataset folder structure |
| `mmrotate/mmrotate/core/bbox/coder/delta_midpointoffset_rbbox_coder.py` | `.clamp()`, `.nan_to_num()`, restored `repeat_interleave` | Numerical stability |
| `mmrotate/mmrotate/models/dense_heads/rotated_rpn_head.py` | Return zero-losses when no targets | Prevent crash on empty images |
| `mmrotate/mmrotate/models/roi_heads/bbox_heads/rotated_bbox_head.py` | Use `nonzero()` instead of bool indexing | Cleaner indexing |
| `/tmp/mmcv-1.7.2/mmcv/parallel/_functions.py` | Wrap int device id in `torch.device()` | PyTorch 2.x API compatibility |
| `my_config/...amod.py` | `data_root`, `samples_per_gpu=4`, fp16 | Local paths + performance |
| `yolo/convert_to_yolo_obb.py` | New file | Convert AMOD CSV → YOLO OBB format |
| `yolo/amod_yolo.yaml` | New file | YOLO dataset config |
| `yolo/train_yolo26_obb.py` | New file | YOLO26 training entry point |
| `mmrotate/mmrotate/datasets/amod.py` | `np.array(obb_bboxes).reshape(-1, 5)` | Empty bbox array was 1D `(0,)` → must be 2D `(0, 5)` for eval |
| `mmrotate/mmrotate/core/evaluation/eval_map.py` | Added `ndim` guard before `[gt_inds, :]` | Defensive fix for same 1D array issue |

---

## Dataset Structure Note (V1.0 vs README)

The [AMOD GitHub README](https://github.com/unique-chan/AMOD) shows an older structure:
```
train/train_imgs/0000/EO_0000_0.png
train/train_labels/ANNOTATION-EO_0000_0.csv
```
**V1.0 (what we have) uses a different structure:**
```
data/train/{scene_id}/{angle}/EO_{scene_id}_{angle}.png
data/train/{scene_id}/{angle}/ANNOTATION-EO_{scene_id}_{angle}.csv
```
This is already handled in our patched `mmrotate/mmrotate/datasets/amod.py`.

---

## Notes for Future Sessions

- **mmcv-full 1.7.2 is installed as editable** at `/tmp/mmcv-1.7.2` — if `/tmp` is cleared, rebuild with:
  ```bash
  cd /tmp/mmcv-1.7.2
  source ~/Repositories/AMOD/venv/bin/activate
  MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation
  ```
- **Resume interrupted OrientedRCNN training** (e.g. after reboot):
  ```bash
  python mmrotate/tools/train.py \
    "my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py" \
    --work-dir work_dirs/orientedrcnn_swinS_baseline \
    --resume-from work_dirs/orientedrcnn_swinS_baseline/latest.pth
  ```
  `resume-from` restores weights + optimizer + epoch. `load-from` only loads weights.
- **YOLO label files** already generated at `data/{split}/{scene}/{angle}/*.txt`
- **Checkpoints** saved every 5 epochs to `work_dirs/orientedrcnn_swinS_baseline/` (e.g. `epoch_5.pth`, `epoch_10.pth`, ...) plus `best_mAP.pth` when a new best is reached
- **val_mini.txt** — 170 randomly sampled scenes from val.txt (seed=42), used during training to avoid slow full-set evaluation. Final paper numbers use full `val.txt` (run test.py manually)
- The dataset has **corrupt/out-of-bounds labels** in a small number of images — YOLO skips them automatically with a warning; OrientedRCNN handles them via the custom `amod.py` loader
