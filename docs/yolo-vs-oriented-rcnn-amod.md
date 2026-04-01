---
title: "Two-Stage vs Single-Stage Oriented Detection on Synthetic Aerial Imagery"
slug: "yolo-vs-oriented-rcnn-amod"
summary: "I ran a three-way comparison of Oriented R-CNN, YOLO11s-OBB, and YOLO26s-OBB on the AMOD synthetic aerial dataset. YOLO26s tops the chart at mAP@50 0.934 and ~400 FPS, but the per-class breakdown tells a more interesting story than the headline number."
published_at: "2026-03-31"
category: "engineering"
tags: ["ai", "engineering"]
---

![AMOD example scene](/docs/figures/fig_header.png)

---

If you have ever played ArmA 3, you know two things: it is the most realistic military simulator ever made, and at some point a friendly soldier will get run over by his own jeep in a flat open field with no enemies nearby. The game is a goldmine for emergent chaos. It is also, apparently, a goldmine for aerial object detection training data.

I also recently got an RTX 5090 laptop. A 24 GB GPU sitting mostly idle while I write code felt like a personal insult, so I decided to give it something to think about. Training three object detection models back to back for a total of ~38 hours seemed like a reasonable punishment for the card, and a good excuse to finally run an experiment I had been putting off. I also wanted to try YOLO26, which came out in January 2026, about three months ago, and already has oriented bounding box support. YOLO26 introduces a proper angle loss and an NMS-free OBB head, so I was curious whether those new 2026 features actually moved the needle compared to YOLO11.

So I ran a comparison of three object detection models on the same dataset to settle a question I had: is a heavy two-stage detector actually more accurate than a lightweight single-stage one on aerial imagery, or does the efficiency gap come for free?

The short answer is that YOLO26s-OBB tops the chart at mAP@50 0.934 (~400 FPS), followed by YOLO11s at 0.904, while Oriented R-CNN trails at 0.895 at roughly 7x the parameter count. But the per-class breakdown tells a more nuanced story, and the results come with a significant caveat: this is synthetic data from a video game, and the high accuracy numbers reflect that.

One important note upfront: the Oriented R-CNN implementation I used was not built from scratch. It came directly from the [original AMOD repository](https://github.com/unique-chan/AMOD), which provides a config and codebase specifically targeting this dataset. My contribution was adapting it to run on modern hardware and comparing it against both YOLO variants under matched conditions.

All training configs, raw logs, and reproducibility notes are in [the experiment repository](https://github.com/flyingraptor/AMOD-detection-comparison).

> **Note on synthetic data.** The AMOD dataset is rendered in ArmA 3, not collected from real sensors. Uniform lighting, clean backgrounds, no motion blur, no sensor noise. The mAP numbers here are not representative of real-world aerial detection performance. This is a study of model training dynamics and architectural trade-offs in a controlled environment.

![Detection example 1](/docs/figures/fig_detection_ex1.png)
![Detection example 2](/docs/figures/fig_detection_ex2.png)
![Detection example 3](/docs/figures/fig_detection_ex3.png)
*Three scenes from the AMOD validation set with oriented bounding box predictions. Oblique viewing angles, mixed terrain, different object densities.*

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

The implementation comes from the [original AMOD repository](https://github.com/unique-chan/AMOD) by unique-chan, which ships a ready-made MMRotate 0.3.4 config targeting this dataset specifically. I did not implement the model from scratch; I adapted it to run on modern hardware. The config required several changes to run on my machine (RTX 5090 24 GB, Intel Core Ultra 9 275HX, 64 GB RAM, Ubuntu Linux), listed in the [experiment repository](https://github.com/flyingraptor/AMOD-detection-comparison):

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

A single-stage, anchor-free detector. It predicts oriented boxes directly from feature maps in one forward pass, no region proposals needed. The architecture uses C3k2 blocks for the backbone and a C2PSA neck for spatial attention. YOLO11 still requires NMS as a post-processing step at inference. End-to-end NMS-free inference was pioneered earlier in YOLOv10 and is central to YOLO26's design, but YOLO11 does not include it.

![YOLO11s-OBB pipeline](/docs/figures/fig_yolo_pipeline.png)

Key numbers: 9.7M parameters and 57.1 GFLOPs at the 1024px resolution used here (22.3 GFLOPs at the standard 640px benchmark resolution). That is about 7x fewer parameters than Oriented R-CNN.

Training setup was matched to Oriented R-CNN where possible: 30 epochs, batch size 4, 1024px images, same optimizer and initial learning rate, same augmentation (rotation and flips), same validation scenes for intermediate evaluation.

### YOLO26s-OBB

The latest model in the YOLO family at the time of this experiment. YOLO26 was released on January 14, 2026, roughly three months before this experiment. It follows YOLO12 in the Ultralytics lineage and introduces several architectural changes, four of which are directly relevant to oriented detection:

- **DFL removal:** Distribution Focal Loss, present in YOLO11 and earlier, is dropped entirely. This simplifies the inference graph and broadens hardware compatibility, particularly for edge and CPU deployments.
- **End-to-end NMS-free inference:** predictions produced directly via a one-to-one head, no post-processing NMS step required.
- **Dedicated angle loss:** an explicit `angle_loss` term in the training objective, designed to resolve the 0°/180° boundary discontinuity that affects YOLO11 and earlier OBB models.
- **Refined OBB decoding:** optimized decoding in the OBB head paired with the angle loss.

Model stats: 10.5M parameters, 55.1 GFLOPs at 1024px (post-fuse, from Ultralytics docs); the pre-fuse model summary reports 24.5 GFLOPs. Marginally larger than YOLO11s but in the same weight class. Epochs, batch size, optimizer, learning rate, and augmentation were all kept identical to YOLO11s for a direct comparison. We used standard SGD rather than YOLO26's native MuSGD optimizer to keep the training conditions matched.

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
| **YOLO26s-OBB** | **0.934** | **0.714** | **0.920** | **0.868** | **~400** | **~5** | **10.5** | **24.5** |

MMRotate's evaluation script reports mAP@50 only, so mAP@50:95, precision, and recall are not available for Oriented R-CNN. Inference speed was not benchmarked for R-CNN. YOLO FPS figures come from the Ultralytics `.val()` run on an RTX 5090 24 GB, Intel Core Ultra 9 275HX (24 cores), 64 GB RAM, Linux: 3.9 ms per image for YOLO11s (~256 FPS), 2.5 ms per image for YOLO26s (~400 FPS). GFLOPs for YOLO11s is the pre-fuse model summary at 640px (57.1 at 1024px per Ultralytics docs); GFLOPs for YOLO26s is the pre-fuse model summary at 1024px (55.1 post-fuse at 1024px per Ultralytics docs).

![Training loss curves](/docs/figures/fig_loss_curves.png)
*Training loss curves for all three models over 30 epochs.*

![Validation mAP per epoch](/docs/figures/fig_map_curves.png)
*Validation mAP@50 per epoch. All three models converge, but both YOLO variants reach their peak faster than R-CNN.*

### Per-class AP

The headline number hides more than it reveals. Here is the full breakdown:

| Class | Oriented R-CNN | YOLO11s | YOLO26s | R-CNN vs YOLO11s | YOLO26s vs YOLO11s |
|---|---|---|---|---|---|
| Armored (27.3%) | 0.9005 | 0.890 | **0.932** | -0.011 | **+0.042** |
| Artillery (0.9%) | 0.9036 | 0.906 | **0.990** | +0.002 | **+0.084** |
| Helicopter (8.2%) | 0.7946 | **0.967** | **0.970** | **+0.172** | +0.003 |
| LCU (0.5%) | 0.9091 | **0.983** | **0.995** | **+0.074** | +0.012 |
| MLRS (2.2%) | **0.9076** | 0.804 | **0.919** | **-0.104** | **+0.115** |
| Plane (7.3%) | 0.9085 | **0.994** | **0.994** | **+0.086** | 0.000 |
| RADAR (5.0%) | 0.8957 | 0.927 | **0.929** | +0.031 | +0.002 |
| SAM (5.0%) | 0.9025 | 0.937 | **0.938** | +0.035 | +0.001 |
| Self-prop. Artillery (1.8%) | **0.9068** | 0.690 | 0.677 | **-0.217** | -0.013 |
| Support (16.4%) | **0.9055** | 0.850 | **0.913** | -0.056 | **+0.063** |
| Tank (15.9%) | 0.9053 | **0.954** | **0.963** | +0.049 | +0.009 |
| TEL (1.3%) | 0.9034 | 0.943 | **0.983** | +0.040 | **+0.040** |
| **mAP@50** | 0.8952 | 0.9040 | **0.934** | +0.009 | **+0.030** |

![Per-class AP](/docs/figures/fig_per_class_ap.png)
*Per-class AP@50 for all three models. The gap on Helicopter and Self-propelled Artillery stands out.*

YOLO11s wins on 8 of 12 classes against R-CNN. The biggest advantages are on Helicopter (+0.172), Plane (+0.086), and LCU (+0.074). These are classes with distinctive, rigid silhouettes. An anchor-free head with rotation augmentation handles those shapes well.

Oriented R-CNN holds a meaningful lead on Self-propelled Artillery (-0.217 for YOLO11s), MLRS (-0.104), and Support (-0.056). Self-propelled Artillery and MLRS are both low-frequency classes (1.8% and 2.2% of instances) with irregular, visually complex shapes. The two-stage proposal mechanism gives R-CNN more precise localisation of ambiguous targets. This is the strongest argument for keeping R-CNN if your application requires high recall on rare, hard categories.

The Helicopter result deserves a note. R-CNN's lowest AP across all classes is Helicopter at 0.7946, well below its 0.895-0.909 range on other classes. At shallow oblique angles (40°-50°), rotor blades and fuselage produce elongated OBB aspect ratios that are hard for a region proposal network to regress accurately. YOLO11s's anchor-free head, combined with wide rotation augmentation, handles this without the intermediate proposal step.

YOLO26s improves on YOLO11s across 10 of 12 classes, with the biggest gains on MLRS (+0.115), Artillery (+0.084), Support (+0.063), and Armored (+0.042). The dedicated angle loss in YOLO26 appears most effective on classes with complex or elongated OBB orientations. On Helicopter the margin is nearly flat (+0.003), which suggests YOLO11s's anchor-free head was already handling that shape well without an explicit angle term. The one regression is Self-propelled Artillery (-0.013), which remains the hardest class for both YOLO variants. At 1.8% of training instances and irregular silhouettes, neither single-stage model reliably closes the gap with R-CNN on that class.

---

## What This Means in Practice

If you are iterating quickly and need a general-purpose aerial detector: both YOLO variants train in around 5 hours instead of 28, and YOLO26s runs at ~400 FPS while beating a significantly larger model on overall accuracy. YOLO26s's dedicated angle loss gives it a consistent edge over YOLO11s across most classes with no additional training cost. There is no meaningful trade-off at the aggregate level.

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
