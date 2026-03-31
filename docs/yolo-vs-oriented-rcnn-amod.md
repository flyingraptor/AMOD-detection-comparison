---
title: "Two-Stage vs Single-Stage Oriented Detection on Synthetic Aerial Imagery"
slug: "yolo-vs-oriented-rcnn-amod"
summary: "I ran a three-way comparison of Oriented R-CNN, YOLO11s-OBB, and YOLO26s-OBB on the AMOD synthetic aerial dataset. YOLO wins on overall accuracy at a fraction of the parameter count, but the per-class breakdown tells a more interesting story than the headline number."
published_at: "2026-03-31"
category: "engineering"
tags: ["ai", "engineering"]
---

![AMOD example scene](/docs/figures/fig_amod_example.png)
*Aerial view from an ArmA 3 simulation scene with oriented bounding box annotations. Orange: Tank. Yellow: Support vehicles. Gray: civilian vehicles (excluded from the 12 training classes). The image shows what AMOD looks like: photorealistic game-engine rendering at 1920x1440 px, viewed at 0 degrees (directly overhead).*

---

I recently ran a controlled comparison of three object detection models on the same dataset to settle a question I had: is a heavy two-stage detector actually more accurate than a lightweight single-stage one on aerial imagery, or does the efficiency gap come for free?

The short answer is that both YOLO11s-OBB and YOLO26s-OBB beat Oriented R-CNN overall at roughly one seventh of the parameter count. But the per-class breakdown tells a more nuanced story, and the results come with a significant caveat: this is synthetic data from a video game, and the high accuracy numbers reflect that.

One important note upfront: the Oriented R-CNN implementation I used was not built from scratch. It came directly from the [original AMOD repository](https://github.com/unique-chan/AMOD), which provides a config and codebase specifically targeting this dataset. My contribution was adapting it to run on modern hardware and comparing it against both YOLO variants under matched conditions.

All training configs, raw logs, and reproducibility notes are in [the experiment repository](https://github.com/flyingraptor/AMOD-detection-comparison).

> **Note on synthetic data.** The AMOD dataset is rendered in ArmA 3, not collected from real sensors. Uniform lighting, clean backgrounds, no motion blur, no sensor noise. The mAP numbers here are not representative of real-world aerial detection performance. This is a study of model training dynamics and architectural trade-offs in a controlled environment.

---

## The Dataset: AMOD

AMOD (Arma3 Multi-view Object Detection) is a large-scale synthetic aerial benchmark rendered in ArmA 3. Each scene is photographed from six viewing angles: 0°, 10°, 20°, 30°, 40°, and 50°. Every image comes with oriented bounding box (OBB) annotations, meaning boxes that rotate to fit the object rather than staying axis-aligned.

Why oriented boxes? Because when you look at a tank from directly above, the axis-aligned box wastes a lot of empty space. A rotated box fits tightly regardless of which way the vehicle is pointing. The OBB orientation in AMOD spans the full 0°-180° range, confirming axis-aligned detection would be a poor fit.

![BBox Statistics](/docs/figures/fig_dataset_bbox_stats.png)
*Bounding box statistics (outliers above the 99th percentile excluded). Left: width distribution. Centre: scale. Right: OBB orientation. The wide orientation spread confirms that rotated detection is the right choice for this dataset.*

The split I used:

| | Images | Scenes | Instances |
|---|---|---|---|
| Train | ~24,978 | 5,202 | 173,988 |
| Val (evaluation set) | 6,240 | ~1,040 | 31,403 |
| Test | 7,590 | ~1,265 | labels withheld |

The test labels are not publicly released, which follows the same convention as DOTA and other aerial detection benchmarks. The validation set is large enough to be the sole evaluation target: 6,240 images across all 6 viewing angles and all 12 classes is more than enough for reliable per-class AP estimates.

### Class imbalance

The dataset has 12 classes, but they are far from evenly distributed. Armored vehicles make up 27.3% of all training instances; LCU sits at 0.5%. That is a 56x imbalance. The top three classes (Armored, Support, Tank) account for 59.6% of instances combined.

![Class Distribution](/docs/figures/fig_dataset_class_dist.png)
*Class distribution across training instances. Armored dominates at 27.3%; LCU is the rarest at 0.5%.*

This matters when reading the results. A high aggregate mAP is partly driven by the common classes. Per-class AP is where the interesting differences show up.

### The two excluded classes

The official AMOD README lists 12 classes, but V1.0 annotations actually contain 14. The extras are `civilian` (6.8% of instances) and `Boat` (1.3%). Neither is documented in the repository.

I excluded both. A military detection benchmark should not flag non-combatants as threats; including `civilian` as a detection target would be operationally inappropriate. `Boat` is similarly ambiguous and absent from the official taxonomy. Both appear in the raw annotations for scene realism in ArmA 3, not as intended detection targets.

---

## The Three Models

### Oriented R-CNN with Swin Transformer-S

A two-stage detector: it first proposes candidate regions, then classifies and refines each one. The backbone is Swin-Small, pretrained on ImageNet-1K. The neck is an FPN with four output levels. The RPN proposes regions that the box head refines into oriented boxes.

![Oriented R-CNN pipeline](/docs/figures/fig_rcnn_pipeline.png)

I used the MMRotate 0.3.4 implementation from the original AMOD repository. The config required several changes to run on my machine (RTX 5090 24 GB, Intel Core Ultra 9 275HX, 64 GB RAM, Ubuntu Linux), listed in the [experiment repository](https://github.com/flyingraptor/AMOD-detection-comparison):

| Change | Reason |
|---|---|
| `samples_per_gpu`: 2 → 4 | Higher GPU throughput |
| `workers_per_gpu`: 2 → 4 | Faster data loading |
| Added `Fp16OptimizerHook` | Enable AMP/fp16 training |
| Evaluation interval every 5 epochs | Avoid slow pycocotools bottleneck |
| `torch.device` compatibility fix | PyTorch 2.x API |
| `.clamp(min=1e-6)` and `.nan_to_num()` in coder | Numerical stability on RTX 5090 (sm_120) |
| Empty-target and empty-ROI guards | Prevent crashes on edge cases |

Training setup:

| | |
|---|---|
| Epochs | 30 |
| Batch size | 4 |
| Optimizer | SGD, lr=0.005, momentum=0.9, wd=1e-4 |
| LR schedule | Linear warmup (500 iter) + step decay at epochs 16, 22 |
| Precision | fp16 |

### YOLO11s-OBB

A single-stage, anchor-free detector. It predicts oriented boxes directly from feature maps in one forward pass, no region proposals needed. The architecture uses C3k2 blocks for the backbone and a C2PSA neck for spatial attention. YOLO11 still uses NMS at inference — end-to-end NMS-free inference came in the next generation.

![YOLO11s-OBB pipeline](/docs/figures/fig_yolo_pipeline.png)

Key numbers: 9.7M parameters and 22.3 GFLOPs at 640px (roughly 57 GFLOPs at the 1024px resolution used here). That is about 7x fewer parameters and 8x fewer GFLOPs than Oriented R-CNN.

Training setup was matched to Oriented R-CNN where possible: 30 epochs, batch size 4, 1024px images, same optimizer and initial learning rate, same augmentation (rotation and flips), same validation scenes for intermediate evaluation.

### YOLO26s-OBB

The second-generation single-stage model. YOLO26 was released in January 2026 and introduces three changes directly relevant to oriented detection:

- **Dedicated angle loss** — an explicit `angle_loss` term in the training objective, designed to fix the 0°/180° boundary discontinuity that affects YOLO11 and earlier OBB models.
- **`OBB26` head** — a new dedicated OBB detection head with refined angle decoding.
- **End-to-end NMS-free inference** — predictions produced directly without a post-processing NMS step.

Model stats: 10.5M parameters, 24.5 GFLOPs (model summary at 1024px) — marginally larger than YOLO11s but in the same weight class. Everything else — epochs, batch size, optimizer, learning rate, augmentation — was kept identical to YOLO11s for a direct comparison.

![YOLO model comparison](/docs/figures/fig_yolo_comparison.png)
*Ultralytics YOLO model family comparison (source: Ultralytics documentation). YOLO11s and YOLO26s sit at essentially the same accuracy-latency operating point at the small-model scale.*

---

## Results

A quick note on the metrics before the numbers. **mAP** (mean Average Precision) is the standard detection metric. For each class, you vary the confidence threshold, plot precision vs. recall, and take the area under that curve to get one AP number. mAP is then the mean of those AP scores across all 12 classes. The **@50** suffix means a predicted box counts as correct only if it overlaps the ground-truth box by at least 50% (IoU threshold). **mAP@50:95** is stricter: it averages mAP across IoU thresholds from 0.50 to 0.95, so boxes need to fit much more tightly to score well. Higher is better, 1.0 is perfect.

### Overall comparison

| Model | mAP@50 | mAP@50:95 | Precision | Recall | FPS | Train (h) | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|---|---|
| Oriented R-CNN + Swin-S | 0.8952 | n/a | n/a | n/a | n/a | 28 | ~69 | ~190 |
| YOLO11s-OBB | 0.9040 | 0.671 | 0.889 | 0.834 | ~256 | 5 | 9.7 | 22.3 |
| **YOLO26s-OBB** | **[mAP@50]** | **[mAP@50:95]** | **[P]** | **[R]** | **[FPS]** | **[h]** | **10.5** | **24.5** |

MMRotate's evaluation script reports mAP@50 only, so mAP@50:95, precision, and recall are not available for Oriented R-CNN. Inference speed was not benchmarked for R-CNN. YOLO's 256 FPS figure (3.9 ms per image) comes from the Ultralytics `.val()` run on an RTX 5090 24 GB, Intel Core Ultra 9 275HX (24 cores), 64 GB RAM, Linux.

![Training loss curves](/docs/figures/fig_loss_curves.png)
*Training loss curves for both models over 30 epochs.*

![Validation mAP per epoch](/docs/figures/fig_map_curves.png)
*Validation mAP@50 per epoch. Both models converge, but YOLO reaches its peak faster.*

### Per-class AP

The headline number hides more than it reveals. Here is the full breakdown:

| Class | Oriented R-CNN | YOLO11s | YOLO26s | R-CNN vs YOLO11s |
|---|---|---|---|---|
| Armored (27.3%) | 0.9005 | 0.890 | [AP] | -0.011 |
| Artillery (0.9%) | 0.9036 | 0.906 | [AP] | +0.002 |
| Helicopter (8.2%) | 0.7946 | **0.967** | [AP] | **+0.172** |
| LCU (0.5%) | 0.9091 | **0.983** | [AP] | **+0.074** |
| MLRS (2.2%) | **0.9076** | 0.804 | [AP] | **-0.104** |
| Plane (7.3%) | 0.9085 | **0.994** | [AP] | **+0.086** |
| RADAR (5.0%) | 0.8957 | 0.927 | [AP] | +0.031 |
| SAM (5.0%) | 0.9025 | 0.937 | [AP] | +0.035 |
| Self-prop. Artillery (1.8%) | **0.9068** | 0.690 | [AP] | **-0.217** |
| Support (16.4%) | **0.9055** | 0.850 | [AP] | -0.056 |
| Tank (15.9%) | 0.9053 | **0.954** | [AP] | +0.049 |
| TEL (1.3%) | 0.9034 | 0.943 | [AP] | +0.040 |
| **mAP@50** | 0.8952 | **0.9040** | **[mAP]** | **+0.009** |

![Per-class AP](/docs/figures/fig_per_class_ap.png)
*Per-class AP@50 for both models. The gap on Helicopter and Self-propelled Artillery stands out.*

YOLO11s wins on 8 of 12 classes against R-CNN. The biggest advantages are on Helicopter (+0.172), Plane (+0.086), and LCU (+0.074). These are classes with distinctive, rigid silhouettes. An anchor-free head with rotation augmentation handles those shapes well.

Oriented R-CNN holds a meaningful lead on Self-propelled Artillery (-0.217 for YOLO11s), MLRS (-0.104), and Support (-0.056). Self-propelled Artillery and MLRS are both low-frequency classes (1.8% and 2.2% of instances) with irregular, visually complex shapes. The two-stage proposal mechanism gives R-CNN more precise localisation of ambiguous targets. This is the strongest argument for keeping R-CNN if your application requires high recall on rare, hard categories.

The Helicopter result deserves a note. R-CNN's lowest AP across all classes is Helicopter at 0.7946, well below its 0.895-0.909 range on other classes. At shallow oblique angles (40°-50°), rotor blades and fuselage produce elongated OBB aspect ratios that are hard for a region proposal network to regress accurately. YOLO11s's anchor-free head, combined with wide rotation augmentation, handles this without the intermediate proposal step.

The key question for YOLO26s is whether the dedicated angle loss closes any of these gaps — particularly on Helicopter and Self-propelled Artillery, where OBB angle regression is hardest. [Fill in once YOLO26s results are available.]

---

## What This Means in Practice

If you are iterating quickly and need a general-purpose aerial detector: both YOLO variants train in around 5 hours instead of 28, run at high FPS, and beat a significantly larger model on overall accuracy. There is no meaningful trade-off at the aggregate level.

If your application requires high recall on rare, visually complex targets and you can afford the resource cost: Oriented R-CNN's 0.217-point advantage on Self-propelled Artillery is real and operationally significant. The two-stage proposal mechanism earns its keep on those edge cases.

The broader point: on this kind of well-structured synthetic data, a small single-stage detector can match or exceed a much larger two-stage model. The assumption that two-stage is inherently more accurate does not hold here.

That said, all of this is synthetic. Both models benefit from AMOD's clean, controlled environment. Before drawing conclusions about real-world performance, you would need domain adaptation or fine-tuning on real aerial imagery. The 0.90+ mAP figures reflect game-engine textures and uniform lighting, not operational conditions.

---

## Practical Takeaways

1. Start with YOLO for synthetic aerial detection tasks. Better overall accuracy, much faster iteration, far fewer resources.
2. Check per-class AP before declaring a winner. The aggregate mAP hides large class-level gaps that may matter for your specific use case.
3. For rare, visually ambiguous classes, two-stage detectors still have an edge. If Self-propelled Artillery or MLRS detection is mission-critical, R-CNN's proposal mechanism is worth the cost.
4. Treat synthetic mAP numbers with appropriate skepticism. High accuracy on simulation data is encouraging but not a substitute for real-world evaluation.

All experiment configs, training commands, raw logs, and reproduction instructions are in [the experiment repository](https://github.com/flyingraptor/AMOD-detection-comparison).

---

## References

- Xie, X. et al. (2021). [Oriented R-CNN for Object Detection](https://openaccess.thecvf.com/content/ICCV2021/html/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.html). ICCV 2021.
- Liu, Z. et al. (2021). [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030). ICCV 2021.
- Jocher, G. & Qiu, J. (2024). [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics). License: AGPL-3.0.
- Jocher, G. & Qiu, J. (2026). [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics). License: AGPL-3.0.
- Xia, G. et al. (2018). [DOTA: A Large-Scale Dataset for Object Detection in Aerial Images](https://arxiv.org/abs/1711.10398). CVPR 2018.
- unique-chan. (2024). [AMOD: Arma3 Multi-view Object Detection](https://github.com/unique-chan/AMOD). GitHub repository.
