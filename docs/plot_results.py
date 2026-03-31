"""
Generate publication-ready figures from training logs.
  - Oriented R-CNN : TensorBoard event files in RCNN_LOG
  - YOLO11s-OBB    : TensorBoard event files in YOLO_LOG  (primary)
                     CSV fallback via YOLO_CSV             (if no TB events yet)

Run from the AMOD root directory:
    source venv/bin/activate
    python docs/plot_results.py
Figures are saved to docs/figures/
"""

from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

FIGURES  = Path("docs/figures")
FIGURES.mkdir(exist_ok=True)

RCNN_LOG = "work_dirs/orientedrcnn_swinS_baseline/tf_logs"
YOLO_LOG = "work_dirs/yolo26_obb_baseline/train"
YOLO_CSV = "work_dirs/yolo26_obb_baseline/train/results.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_tb(logdir, scalar_key):
    """Load a scalar from a TensorBoard event file. Returns (steps, vals)."""
    from tensorboard.backend.event_processing import event_accumulator
    p = Path(logdir)
    if not p.exists():
        return [], []
    ea = event_accumulator.EventAccumulator(str(p), size_guidance={"scalars": 0})
    ea.Reload()
    if scalar_key not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(scalar_key)
    return [e.step for e in events], [e.value for e in events]


def load_yolo_csv(col):
    """Load a column from YOLO results.csv. Returns (epochs, values)."""
    p = Path(YOLO_CSV)
    if not p.exists():
        return [], []
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if col in row and row[col]:
                rows.append((int(row["epoch"]), float(row[col])))
    if not rows:
        return [], []
    return list(zip(*rows))


def yolo_series(tb_key, csv_col):
    """Try TensorBoard first, fall back to CSV for YOLO data."""
    steps, vals = load_tb(YOLO_LOG, tb_key)
    if steps:
        return steps, vals
    return load_yolo_csv(csv_col)


def smooth(vals, weight=0.85):
    """Exponential moving average — mirrors TensorBoard smoothing slider."""
    smoothed, last = [], vals[0]
    for v in vals:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


# ── Fig 1: Training loss curves ───────────────────────────────────────────────

def plot_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    # --- RCNN (total loss from TensorBoard) ---
    ax = axes[0]
    steps, vals = load_tb(RCNN_LOG, "train/loss")
    if steps:
        ax.plot(steps, vals, alpha=0.2, color="steelblue", linewidth=0.7)
        ax.plot(steps, smooth(vals), color="steelblue", linewidth=1.8,
                label="Total loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("Oriented R-CNN + Swin-S")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    # --- YOLO (total loss + individual components from TB or CSV) ---
    ax = axes[1]
    comp_keys = [
        ("train/box_loss",   "train/box_loss",   "Box loss",   "steelblue"),
        ("train/cls_loss",   "train/cls_loss",   "Cls loss",   "darkorange"),
        ("train/dfl_loss",   "train/dfl_loss",   "DFL loss",   "seagreen"),
        ("train/angle_loss", "train/angle_loss", "Angle loss", "orchid"),
    ]
    all_epochs, all_arrs = None, []
    for tb_key, csv_col, label, color in comp_keys:
        epochs, vals = yolo_series(tb_key, csv_col)
        if not epochs:
            continue
        all_epochs = epochs
        all_arrs.append(np.array(vals))
        ax.plot(epochs, vals, alpha=0.25, color=color, linewidth=0.7)
        ax.plot(epochs, smooth(vals), color=color, linewidth=1.2,
                linestyle="--", label=label)

    if all_epochs is not None and len(all_arrs) == 4:
        total = sum(all_arrs)
        ax.plot(all_epochs, total, alpha=0.25, color="black", linewidth=0.7)
        ax.plot(all_epochs, smooth(list(total)), color="black",
                linewidth=2.0, label="Total loss")

    if all_epochs is None:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("YOLO11s-OBB")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.suptitle("Training Loss Curves", fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "fig_loss_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Fig 2: mAP50 per epoch — both models ──────────────────────────────────────

RCNN_STEPS_PER_EPOCH = 6062   # 24,248 images / batch 4

def plot_map_curves():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # RCNN — convert steps → epochs
    steps, vals = load_tb(RCNN_LOG, "val/mAP")
    if steps:
        epochs = [s / RCNN_STEPS_PER_EPOCH for s in steps]
        ax.plot(epochs, vals, "o-", color="steelblue", linewidth=1.8,
                markersize=5, label="Oriented R-CNN + Swin-S")

    # YOLO — already in epochs
    steps, vals = yolo_series("metrics/mAP50(B)", "metrics/mAP50(B)")
    if steps:
        ax.plot(steps, vals, "s-", color="darkorange", linewidth=1.8,
                markersize=5, label="YOLO11s-OBB")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50")
    ax.set_title("Validation mAP@50 per Epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIGURES / "fig_map_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Fig 3: mAP50-95 per epoch — both models ───────────────────────────────────

def plot_map5095_curves():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # YOLO only (RCNN doesn't log mAP50-95 during training)
    steps, vals = yolo_series("metrics/mAP50-95(B)", "metrics/mAP50-95(B)")
    if steps:
        ax.plot(steps, vals, "s-", color="darkorange", linewidth=1.8,
                markersize=5, label="YOLO11s-OBB")
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50:95")
    ax.set_title("Validation mAP@50:95 per Epoch — YOLO11s-OBB")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIGURES / "fig_map5095_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Fig 4: OrientedRCNN mAP progression with LR annotations ──────────────────

def plot_rcnn_map_progression():
    steps, vals = load_tb(RCNN_LOG, "val/mAP")
    if not steps:
        print("  [skip] No val/mAP in RCNN logs yet.")
        return

    epochs = [s / RCNN_STEPS_PER_EPOCH for s in steps]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, vals, "o-", color="steelblue", linewidth=2,
             markersize=6, label="mAP@50 (val_mini)")
    for e, v in zip(epochs, vals):
        ax1.annotate(f"{v:.4f}", (e, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    lr_steps, lr_vals = load_tb(RCNN_LOG, "learning_rate")
    if lr_steps:
        lr_epochs = [s / RCNN_STEPS_PER_EPOCH for s in lr_steps]
        ax2 = ax1.twinx()
        ax2.plot(lr_epochs, lr_vals, color="tomato", linewidth=1.2,
                 alpha=0.6, linestyle="--", label="Learning rate")
        ax2.set_ylabel("Learning Rate", color="tomato", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="tomato")
        ax2.set_yscale("log")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP@50")
    ax1.set_title("Oriented R-CNN + Swin-S — mAP Progression & LR Schedule")
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    out = FIGURES / "fig_rcnn_map_progression.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Fig 5: YOLO mAP progression with LR ──────────────────────────────────────

def plot_yolo_map_progression():
    epochs, vals = yolo_series("metrics/mAP50(B)", "metrics/mAP50(B)")
    if not epochs:
        print("  [skip] No YOLO mAP data yet.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, vals, "s-", color="darkorange", linewidth=2,
             markersize=6, label="mAP@50 (val_mini)")
    for e, v in zip(epochs, vals):
        ax1.annotate(f"{v:.4f}", (e, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    lr_epochs, lr_vals = yolo_series("lr/pg0", "lr/pg0")
    if lr_epochs:
        ax2 = ax1.twinx()
        ax2.plot(lr_epochs, lr_vals, color="tomato", linewidth=1.2,
                 alpha=0.6, linestyle="--", label="Learning rate")
        ax2.set_ylabel("Learning Rate", color="tomato", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="tomato")
        ax2.set_yscale("log")
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP@50")
    ax1.set_title("YOLO11s-OBB — mAP Progression & LR Schedule")
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    out = FIGURES / "fig_yolo_map_progression.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Fig 6: Per-class AP@50 bar chart (fill after final eval) ─────────────────

def plot_per_class():
    classes = [
        "Armored", "Artillery", "Helicopter", "LCU", "MLRS", "Plane",
        "RADAR", "SAM", "Self-prop. Art.", "Support", "Tank", "TEL",
    ]
    # Oriented R-CNN ep30 on full val (6,246 images) ✅
    rcnn_ap = [0.9005, 0.9036, 0.7946, 0.9091, 0.9076, 0.9085,
               0.8957, 0.9025, 0.9068, 0.9055, 0.9053, 0.9034]
    # YOLO11s ep30 on full val (6,240 images) ✅
    yolo_ap = [0.890, 0.906, 0.967, 0.983, 0.804, 0.994,
               0.927, 0.937, 0.690, 0.850, 0.954, 0.943]

    has_rcnn = None not in rcnn_ap
    has_yolo = None not in yolo_ap

    if not has_rcnn and not has_yolo:
        print("  [skip] Per-class plot: fill in rcnn_ap and yolo_ap arrays.")
        return

    x = np.arange(len(classes))
    width = 0.35 if (has_rcnn and has_yolo) else 0.5
    fig, ax = plt.subplots(figsize=(13, 5))

    if has_rcnn and has_yolo:
        ax.bar(x - width/2, rcnn_ap, width,
               label="Oriented R-CNN + Swin-S", color="steelblue")
        ax.bar(x + width/2, yolo_ap, width,
               label="YOLO11s-OBB", color="darkorange")
    elif has_rcnn:
        ax.bar(x, rcnn_ap, width,
               label="Oriented R-CNN + Swin-S (ep30, val_mini)", color="steelblue")
    else:
        ax.bar(x, yolo_ap, width,
               label="YOLO11s-OBB", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("AP@50")
    ax.set_title("Per-Class AP@50 — Baseline Models")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIGURES / "fig_per_class_ap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Fig 7: Accuracy vs Speed Pareto (fill after all experiments) ─────────────

def plot_pareto():
    results = [
        # label                          mAP@50   FPS
        ("Oriented R-CNN + Swin-S",      None,    None),
        ("YOLO11s-OBB",                  None,    None),
    ]
    ready = [(l, m, f) for l, m, f in results if m is not None]
    if not ready:
        print("  [skip] Pareto plot: fill in mAP and FPS values above.")
        return

    labels, maps, fps = zip(*ready)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fps, maps, s=80, zorder=3)
    for label, x, y in zip(labels, fps, maps):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Inference Speed (FPS)")
    ax.set_ylabel("mAP@50 on AMOD test")
    ax.set_title("Accuracy vs Speed — All Variants")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIGURES / "fig_pareto.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures → docs/figures/")
    plot_loss_curves()
    plot_map_curves()
    plot_map5095_curves()
    plot_rcnn_map_progression()
    plot_yolo_map_progression()
    plot_per_class()
    plot_pareto()
    print("Done.")
