"""
prepare_dataset_for_github.py
==============================
Run this LOCALLY on your machine, after downloading the real Kaggle dataset:
  "Chest X-Ray Images (Pneumonia)" by Paul Mooney
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

WHY THIS SCRIPT EXISTS:
The full dataset is ~5,863 images / ~2GB, which is unwieldy to push to a plain
GitHub repo (no Git LFS needed if we keep it small). This script:
  1. Walks your downloaded chest_xray/ folder (train/val/test x NORMAL/PNEUMONIA)
  2. Randomly samples up to --max_per_class images per class per split
  3. Resizes/re-compresses them (default 300px, JPEG quality 85) to shrink file size
  4. Writes them into a new github_dataset/ folder in the exact structure needed:

      github_dataset/
        train/NORMAL/*.jpg
        train/PNEUMONIA/*.jpg
        val/NORMAL/*.jpg
        val/PNEUMONIA/*.jpg
        test/NORMAL/*.jpg
        test/PNEUMONIA/*.jpg

USAGE:
    pip install pillow
    python prepare_dataset_for_github.py --src chest_xray --dst github_dataset --max_per_class 300

Then:
    cd github_dataset
    git init
    git add .
    git commit -m "chest x-ray subset for CNN training"
    git branch -M main
    git remote add origin https://github.com/<your-username>/<your-repo>.git
    git push -u origin main

(If it's a NEW empty repo, create it first on github.com, then run the commands above.)

Once pushed, just send me the repo URL (e.g. https://github.com/yourname/pneumonia-xray-data)
and I'll pull it directly and retrain on it.

NOTE ON SIZE: with --max_per_class 300 and 300px JPEGs, expect roughly 150-250MB
total, which pushes fine as a normal git repo (GitHub's hard per-file limit is
100MB; this script never produces files anywhere near that). If your repo still
feels too large, lower --max_per_class (even 100-150 per class is a big
improvement over what I currently have).
"""
import argparse
import os
import random
import shutil
from pathlib import Path

from PIL import Image

def process_split(src_split_dir, dst_split_dir, max_per_class, size, quality, seed):
    rng = random.Random(seed)
    for cls in ["NORMAL", "PNEUMONIA"]:
        src_dir = src_split_dir / cls
        dst_dir = dst_split_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"  [skip] {src_dir} does not exist")
            continue

        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(files)
        chosen = files[:max_per_class]

        for f in chosen:
            try:
                im = Image.open(src_dir / f).convert("L")  # grayscale, smaller files
                im.thumbnail((size, size), Image.LANCZOS)
                out_name = Path(f).stem + ".jpg"
                im.save(dst_dir / out_name, "JPEG", quality=quality)
            except Exception as e:
                print(f"  [warn] failed on {f}: {e}")

        print(f"  {src_split_dir.name}/{cls}: {len(chosen)} images -> {dst_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to the extracted Kaggle chest_xray folder")
    ap.add_argument("--dst", default="github_dataset", help="Output folder to create")
    ap.add_argument("--max_per_class", type=int, default=300, help="Max images per class per split")
    ap.add_argument("--size", type=int, default=300, help="Max image dimension after resize")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality (1-95)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        print(f"Removing existing {dst} ...")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for split in ["train", "val", "test"]:
        split_src = src / split
        if not split_src.exists():
            print(f"[skip] {split_src} not found (some Kaggle mirrors omit 'val')")
            continue
        print(f"Processing {split} ...")
        process_split(split_src, dst / split, args.max_per_class, args.size, args.quality, args.seed)

    print("\nDone. Output written to:", dst.resolve())
    print("Next: cd", args.dst, "&& git init && git add . && git commit -m 'xray subset' ...")


if __name__ == "__main__":
    main()
