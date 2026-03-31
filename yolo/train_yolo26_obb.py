"""Train YOLO26-OBB on the AMOD dataset.

Usage:
    python train_yolo26_obb.py [--model yolo26s-obb] [--epochs 30] [--batch 4]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = str(Path(__file__).parent / "amod_yolo.yaml")
WORK_DIR  = str(Path(__file__).parent.parent / "work_dirs" / "yolo26_obb_baseline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolo11s-obb",
                        help="Model variant: yolo11n/s/m/l/x-obb")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch",  type=int, default=4,
                        help="Batch size (images per step)")
    parser.add_argument("--imgsz",  type=int, default=1024,
                        help="Training image size (long edge)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model = YOLO(f"{args.model}.pt")

    model.train(
        data      = DATA_YAML,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,
        project   = WORK_DIR,
        name      = "train",
        exist_ok  = True,
        device    = 0,
        # OBB-specific
        task      = "obb",
        # Training hyps — matched to Oriented R-CNN baseline
        optimizer    = "SGD",
        lr0          = 0.005,    # same as RCNN
        lrf          = 0.01,     # final LR = lr0*lrf = 5e-5 (same as RCNN after 2 step decays)
        momentum     = 0.9,      # same as RCNN
        weight_decay = 0.0001,   # same as RCNN
        warmup_epochs = 3.0,     # YOLO default (RCNN had 500-iter warmup ~0.08 ep)
        cos_lr       = True,     # cosine LR decay (YOLO default; RCNN used step decay)
        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=180.0,           # full rotation aug (aerial imagery)
        flipud=0.5,
        fliplr=0.5,
        # Validation — mini set (1,020 images, same 170 scenes as RCNN mini-val)
        val          = True,
        # Logging / checkpoints
        plots        = True,
        save         = True,
        save_period  = 5,        # save every 5 epochs (same as RCNN) + best/last always saved
    )

    print(f"\nTraining complete. Results saved to {WORK_DIR}/train/")
    print("To evaluate on test set:")
    print(f"  python -c \"from ultralytics import YOLO; "
          f"m=YOLO('{WORK_DIR}/train/weights/best.pt'); "
          f"m.val(data='{DATA_YAML}', split='test', task='obb')\"")


if __name__ == "__main__":
    main()
