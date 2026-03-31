"""Convert the AMOD dataset to YOLO OBB format.

YOLO OBB label format per line:
  class_idx  x1_norm  y1_norm  x2_norm  y2_norm  x3_norm  y3_norm  x4_norm  y4_norm

Coordinates are normalised to [0, 1] by the image dimensions (1920 × 1440).
One .txt label file is written alongside each image, with the same stem.
Three image-list files are also written (used by amod_yolo.yaml):
  data/yolo_train.txt, data/yolo_val.txt, data/yolo_test.txt
"""

import os
import csv
import argparse
from pathlib import Path

# ── dataset constants ──────────────────────────────────────────────────────────
IMG_W, IMG_H = 1920, 1440
MODALITY     = "EO"
IMG_EXT      = "png"
ANGLES       = [0, 10, 20, 30, 40, 50]

CLASSES = [
    "Armored", "Artillery", "Helicopter", "LCU", "MLRS", "Plane",
    "RADAR", "SAM", "Self-propelled Artillery", "Support", "Tank", "TEL",
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

DATA_ROOT = Path(__file__).parent.parent / "data"  # .../AMOD/data/


def convert_split(split_name: str, img_subdir: str) -> list[str]:
    """Convert one split and return the list of absolute image paths."""
    split_file = DATA_ROOT / f"{split_name}.txt"
    scene_ids  = split_file.read_text().splitlines()

    img_paths = []
    skipped   = 0

    for scene_id in scene_ids:
        scene_id = scene_id.strip()
        if not scene_id:
            continue
        for angle in ANGLES:
            img_path = DATA_ROOT / img_subdir / scene_id / str(angle) / \
                       f"{MODALITY}_{scene_id}_{angle}.{IMG_EXT}"
            ann_path = DATA_ROOT / img_subdir / scene_id / str(angle) / \
                       f"ANNOTATION-{MODALITY}_{scene_id}_{angle}.csv"

            if not img_path.exists():
                skipped += 1
                continue

            label_lines = []

            if ann_path.exists():
                with open(ann_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("usable", "T").strip() != "T":
                            continue
                        cls = row.get("main_class", "").strip()
                        if cls not in CLASS2IDX:
                            continue
                        try:
                            coords = [
                                float(row["x1"]), float(row["y1"]),
                                float(row["x2"]), float(row["y2"]),
                                float(row["x3"]), float(row["y3"]),
                                float(row["x4"]), float(row["y4"]),
                            ]
                        except (KeyError, ValueError):
                            continue

                        # normalise
                        norm = [
                            coords[0] / IMG_W, coords[1] / IMG_H,
                            coords[2] / IMG_W, coords[3] / IMG_H,
                            coords[4] / IMG_W, coords[5] / IMG_H,
                            coords[6] / IMG_W, coords[7] / IMG_H,
                        ]
                        label_lines.append(
                            f"{CLASS2IDX[cls]} "
                            + " ".join(f"{v:.6f}" for v in norm)
                        )

            label_path = img_path.with_suffix(".txt")
            label_path.write_text("\n".join(label_lines))

            img_paths.append(str(img_path))

    print(f"  {split_name}: {len(img_paths)} images written "
          f"({skipped} missing skipped)")
    return img_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts only, do not write label files")
    args = parser.parse_args()

    print("Converting AMOD → YOLO OBB format …")

    train_imgs = convert_split("train", "train")
    val_imgs   = convert_split("val",   "train")
    test_imgs  = convert_split("test",  "test")

    for name, paths in [("yolo_train", train_imgs),
                        ("yolo_val",   val_imgs),
                        ("yolo_test",  test_imgs)]:
        out = DATA_ROOT / f"{name}.txt"
        out.write_text("\n".join(paths))
        print(f"  Written {out.name}  ({len(paths)} entries)")

    print("Done.")


if __name__ == "__main__":
    main()
