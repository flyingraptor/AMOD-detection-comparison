"""
AMOD Dataset Analysis — generates statistics and figures for the paper.
Covers: class distribution, bbox sizes, OBB angle distribution,
        per-angle image count, corrupt label report.

Run from the AMOD root directory:
    source venv/bin/activate
    python docs/analyze_dataset.py

Figures saved to docs/figures/
Stats printed to console and saved to docs/dataset_stats.txt
"""

import os
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 150,
})

DATA_ROOT  = Path("data")
FIGURES    = Path("docs/figures")
STATS_FILE = Path("docs/dataset_stats.txt")
FIGURES.mkdir(exist_ok=True)

CLASSES = [
    "Armored", "Artillery", "Helicopter", "LCU", "MLRS", "Plane",
    "RADAR", "SAM", "Self-propelled Artillery", "Support", "Tank", "TEL",
]
ANGLES = [0, 10, 20, 30, 40, 50]


# ── data loading ──────────────────────────────────────────────────────────────

def load_annotations(split="train"):
    """
    Walk data/{split}/{scene}/{angle}/ANNOTATION-*.csv
    Returns list of dicts with keys:
        split, scene, angle, cls, x1,y1,x2,y2,x3,y3,x4,y4, img_w, img_h
    and a list of corrupt file paths.
    """
    rows, corrupt = [], []
    base = DATA_ROOT / split
    if not base.exists():
        print(f"[warn] {base} not found, skipping {split}")
        return rows, corrupt

    for scene_dir in sorted(base.iterdir()):
        if not scene_dir.is_dir():
            continue
        for angle_dir in sorted(scene_dir.iterdir()):
            if not angle_dir.is_dir():
                continue
            angle = int(angle_dir.name)
            for csv_path in angle_dir.glob("ANNOTATION-*.csv"):
                try:
                    with open(csv_path, newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                rows.append({
                                    "split": split,
                                    "scene": scene_dir.name,
                                    "angle": angle,
                                    "cls": row["main_class"].strip(),
                                    "x1": float(row["x1"]), "y1": float(row["y1"]),
                                    "x2": float(row["x2"]), "y2": float(row["y2"]),
                                    "x3": float(row["x3"]), "y3": float(row["y3"]),
                                    "x4": float(row["x4"]), "y4": float(row["y4"]),
                                })
                            except (KeyError, ValueError):
                                corrupt.append(str(csv_path))
                except Exception as e:
                    corrupt.append(f"{csv_path}: {e}")
    return rows, corrupt


def poly_to_obb(x1,y1,x2,y2,x3,y3,x4,y4):
    """Returns (cx, cy, w, h, angle_deg) from 4-corner polygon."""
    pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
    cx, cy = pts.mean(axis=0)
    # long edge
    d01 = np.linalg.norm(pts[1]-pts[0])
    d12 = np.linalg.norm(pts[2]-pts[1])
    if d01 >= d12:
        w, h = d01, d12
        angle = math.degrees(math.atan2(pts[1,1]-pts[0,1], pts[1,0]-pts[0,0]))
    else:
        w, h = d12, d01
        angle = math.degrees(math.atan2(pts[2,1]-pts[1,1], pts[2,0]-pts[1,0]))
    return cx, cy, w, h, angle % 180


# ── analysis ─────────────────────────────────────────────────────────────────

def analyze(splits=("train", "test")):
    all_rows, all_corrupt = [], []
    for split in splits:
        rows, corrupt = load_annotations(split)
        all_rows.extend(rows)
        all_corrupt.extend(corrupt)

    if not all_rows:
        print("No annotations found. Check that data/ folder is populated.")
        return

    lines = []  # for stats file

    def log(s=""):
        print(s)
        lines.append(s)

    # ── basic counts ─────────────────────────────────────────────────────────
    log("=" * 60)
    log("AMOD Dataset Statistics")
    log("=" * 60)

    by_split = defaultdict(list)
    for r in all_rows:
        by_split[r["split"]].append(r)

    for split, rows in by_split.items():
        scenes  = len({r["scene"] for r in rows})
        images  = len({(r["scene"], r["angle"]) for r in rows})
        log(f"\n[{split}]  scenes={scenes}  images={images}  instances={len(rows)}")

    log(f"\nTotal instances : {len(all_rows)}")
    log(f"Corrupt records : {len(all_corrupt)}")
    if all_corrupt:
        log("  (first 5 corrupt files:)")
        for p in all_corrupt[:5]:
            log(f"    {p}")

    # ── class distribution ────────────────────────────────────────────────────
    cls_counts = Counter(r["cls"] for r in all_rows)
    log("\nClass distribution (all splits):")
    for cls in sorted(cls_counts, key=cls_counts.get, reverse=True):
        pct = 100 * cls_counts[cls] / len(all_rows)
        log(f"  {cls:<28} {cls_counts[cls]:>7,}  ({pct:.1f}%)")

    # class imbalance ratio
    counts = list(cls_counts.values())
    imbalance = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    log(f"\nClass imbalance ratio (max/min): {imbalance:.1f}x")
    if imbalance > 10:
        log("  ⚠️  HIGH imbalance — consider class-weighted loss or oversampling")
    elif imbalance > 3:
        log("  ⚠️  MODERATE imbalance")
    else:
        log("  ✅  LOW imbalance")

    # ── per-angle counts ──────────────────────────────────────────────────────
    log("\nImages per viewing angle (train):")
    train_rows = by_split.get("train", [])
    angle_img  = defaultdict(set)
    for r in train_rows:
        angle_img[r["angle"]].add((r["scene"], r["angle"]))
    for a in ANGLES:
        log(f"  {a:>3}° : {len(angle_img[a]):>6} images")

    # ── bbox statistics ───────────────────────────────────────────────────────
    widths, heights, areas, angles_obb = [], [], [], []
    for r in all_rows:
        try:
            _, _, w, h, ang = poly_to_obb(
                r["x1"],r["y1"],r["x2"],r["y2"],
                r["x3"],r["y3"],r["x4"],r["y4"])
            if w > 0 and h > 0:
                widths.append(w); heights.append(h)
                areas.append(w * h); angles_obb.append(ang)
        except Exception:
            pass

    if widths:
        # flag likely corrupt annotations (bbox wider/taller than any real image)
        MAX_PLAUSIBLE = 4096
        bad_w = sum(1 for w in widths if w > MAX_PLAUSIBLE)
        bad_h = sum(1 for h in heights if h > MAX_PLAUSIBLE)
        clean_w = [w for w in widths if w <= MAX_PLAUSIBLE]
        clean_h = [h for h in heights if h <= MAX_PLAUSIBLE]
        clean_a = [a for w,h,a in zip(widths,heights,areas)
                   if w <= MAX_PLAUSIBLE and h <= MAX_PLAUSIBLE]

        log(f"\nBounding box sizes (pixels, after removing >{MAX_PLAUSIBLE}px outliers):")
        log(f"  Outlier annotations (w or h > {MAX_PLAUSIBLE}px): {bad_w + bad_h} "
            f"({100*(bad_w+bad_h)/len(widths):.2f}%)")
        log(f"  Width  — mean={np.mean(clean_w):.1f}  median={np.median(clean_w):.1f}  "
            f"min={np.min(clean_w):.1f}  max={np.max(clean_w):.1f}")
        log(f"  Height — mean={np.mean(clean_h):.1f}  median={np.median(clean_h):.1f}  "
            f"min={np.min(clean_h):.1f}  max={np.max(clean_h):.1f}")
        log(f"  Area   — mean={np.mean(clean_a):.0f}  median={np.median(clean_a):.0f}")
        small = sum(1 for a in clean_a if a < 32*32)
        log(f"  Small objects (<32×32 px): {small} ({100*small/len(clean_a):.1f}%)")

    # ── save stats ────────────────────────────────────────────────────────────
    STATS_FILE.write_text("\n".join(lines))
    log(f"\nStats saved → {STATS_FILE}")

    # ── figures ───────────────────────────────────────────────────────────────
    _fig_class_distribution(cls_counts)
    if widths:
        _fig_bbox_stats(widths, heights, areas, angles_obb)
    _fig_angle_distribution(train_rows)


# ── figure helpers ────────────────────────────────────────────────────────────

def _fig_class_distribution(cls_counts):
    labels = sorted(cls_counts, key=cls_counts.get, reverse=True)
    values = [cls_counts[l] for l in labels]
    colors = plt.cm.tab20.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Instance count")
    ax.set_title("Class Distribution — AMOD (all splits)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    # annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.005,
                f"{val:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIGURES / "fig_dataset_class_dist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def _fig_bbox_stats(widths, heights, areas, angles_obb):
    # remove outliers before plotting (keep 99th percentile)
    p99_w = np.percentile(widths, 99)
    p99_a = np.percentile(np.sqrt(areas), 99)
    clean_w = [w for w in widths if w <= p99_w]
    clean_sqrta = [math.sqrt(a) for w, a in zip(widths, areas) if w <= p99_w]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(clean_w, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("Width (px)"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"BBox Width Distribution\n(99th pct ≤ {p99_w:.0f}px, outliers excluded)")

    axes[1].hist(clean_sqrta, bins=60, color="darkorange", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("√Area (px)"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"BBox Scale Distribution (√Area)\n(99th pct ≤ {p99_a:.0f}px, outliers excluded)")

    axes[2].hist(angles_obb, bins=36, range=(0, 180),
                 color="seagreen", edgecolor="none", alpha=0.8)
    axes[2].set_xlabel("OBB angle (°)"); axes[2].set_ylabel("Count")
    axes[2].set_title("OBB Orientation Distribution")

    fig.suptitle("AMOD Bounding Box Statistics", fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "fig_dataset_bbox_stats.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def _fig_angle_distribution(train_rows):
    angle_cls = defaultdict(Counter)
    for r in train_rows:
        angle_cls[r["angle"]][r["cls"]] += 1

    angles = sorted(angle_cls.keys())
    cls_list = sorted({r["cls"] for r in train_rows})
    data = np.array([[angle_cls[a].get(c, 0) for c in cls_list] for a in angles])

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(angles))
    colors = plt.cm.tab20.colors
    for i, cls in enumerate(cls_list):
        ax.bar([str(a)+"°" for a in angles], data[:, i],
               bottom=bottom, label=cls, color=colors[i % 20])
        bottom += data[:, i]
    ax.set_xlabel("Viewing angle"); ax.set_ylabel("Instance count")
    ax.set_title("Instance Count per Viewing Angle (train)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    out = FIGURES / "fig_dataset_angle_dist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    analyze(splits=("train", "test"))
