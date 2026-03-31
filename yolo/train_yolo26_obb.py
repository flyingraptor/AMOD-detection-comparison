"""Train YOLO26-OBB on the AMOD dataset.

YOLO26 key differences vs YOLO11:
  - End-to-end NMS-free inference (dual-head: one-to-one + one-to-many)
  - Specialized angle loss to resolve OBB boundary discontinuity
  - MuSGD optimizer available (hybrid SGD+Muon)
  - DFL removed — simpler export, broader edge compatibility
  - ~55 GFLOPs at 1024px vs ~22 GFLOPs for YOLO11s

For fair comparison with YOLO11s and Oriented R-CNN, core hyperparameters
(lr, momentum, weight_decay, epochs, batch, imgsz) are matched.
MuSGD is available via --optimizer MuSGD if you want to test YOLO26's native setting.

Usage:
    python train_yolo26_obb.py [--model yolo26s-obb] [--epochs 30] [--batch 4]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = str(Path(__file__).parent / "amod_yolo.yaml")
WORK_DIR  = str(Path(__file__).parent.parent / "runs" / "yolo_obb" / "yolo26s_baseline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="yolo26s-obb",
                        help="Model variant: yolo26n/s/m/l/x-obb")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch",     type=int, default=4,
                        help="Batch size (images per step)")
    parser.add_argument("--imgsz",     type=int, default=1024,
                        help="Training image size (long edge)")
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--optimizer", default="SGD",
                        help="SGD (fair comparison) or MuSGD (YOLO26 native)")
    parser.add_argument("--name",      default="train",
                        help="Run name inside project dir")
    args = parser.parse_args()

    model = YOLO(f"{args.model}.pt")

    model.train(
        data      = DATA_YAML,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,
        project   = WORK_DIR,
        name      = args.name,
        exist_ok  = True,
        device    = 0,
        # OBB-specific
        task      = "obb",
        # Training hyps — matched to YOLO11s / Oriented R-CNN baseline for fair comparison
        optimizer    = args.optimizer,
        lr0          = 0.005,
        lrf          = 0.01,     # final LR = lr0 * lrf = 5e-5
        momentum     = 0.9,
        weight_decay = 0.0001,
        warmup_epochs = 3.0,
        cos_lr       = True,
        # Augmentation — same as YOLO11s
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=180.0,           # full rotation aug (aerial imagery)
        flipud=0.5,
        fliplr=0.5,
        # Validation
        val          = True,
        # Logging / checkpoints
        plots        = True,
        save         = True,
        save_period  = 5,
    )

    out = f"{WORK_DIR}/{args.name}"
    print(f"\nTraining complete. Results saved to {out}/")
    print(f"To evaluate on val set:")
    print(f"  python -c \"from ultralytics import YOLO; "
          f"m=YOLO('{out}/weights/best.pt'); "
          f"m.val(data='{DATA_YAML}', imgsz={args.imgsz}, batch={args.batch}, split='val', task='obb')\"")


if __name__ == "__main__":
    main()
