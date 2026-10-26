"""Convert COCO instance annotations to YOLO detection data (apple only)."""
import argparse
import json
import random
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="COCO JSON with images / annotations / categories")
    parser.add_argument("--images", required=True, help="Directory containing source images")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    a = parser.parse_args()
    if a.val_ratio < 0 or a.test_ratio < 0 or a.val_ratio + a.test_ratio >= 1:
        raise ValueError("Invalid split ratios")
    data = json.loads(Path(a.coco).read_text(encoding="utf-8"))
    cats = [c["id"] for c in data["categories"] if "apple" in c["name"].lower()]
    if not cats:
        raise ValueError("No 'apple' category found in COCO categories")
    image_dir = Path(a.images)
    out = Path(a.out).resolve()
    image_records = sorted(data["images"], key=lambda i: str(i["file_name"]))
    annotations = {}
    for ann in data["annotations"]:
        if ann.get("category_id") in cats and not ann.get("ignore", 0):
            annotations.setdefault(ann["image_id"], []).append(ann)
    rng = random.Random(a.seed)
    rng.shuffle(image_records)
    n = len(image_records)
    n_test = int(n * a.test_ratio)
    n_val = int(n * a.val_ratio)
    splits = {"test": image_records[:n_test],
              "val": image_records[n_test:n_test+n_val],
              "train": image_records[n_test+n_val:]}
    counts = {}
    for split, records in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        count = 0
        for item in records:
            # Keep paths relative and unique by image id, to avoid basename collisions.
            src = image_dir / item["file_name"]
            if not src.is_file():
                src = image_dir / Path(item["file_name"]).name
            if not src.is_file():
                raise FileNotFoundError(f"Missing image {item['file_name']} in {image_dir}")
            w, h = float(item["width"]), float(item["height"])
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid dimensions for {src}")
            dest_stem = f"{item['id']}_{src.stem}"
            dst = out / "images" / split / (dest_stem + src.suffix.lower())
            shutil.copy2(src, dst)
            lines = []
            for ann in annotations.get(item["id"], []):
                x, y, bw, bh = map(float, ann["bbox"])
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)
                if x2 <= x1 or y2 <= y1:
                    continue
                xc = (x1+x2)/(2*w)
                yc = (y1+y2)/(2*h)
                nw, nh = (x2-x1)/w, (y2-y1)/h
                lines.append(f"0 {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")
            (out / "labels" / split / (dest_stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            count += len(lines)
        counts[split] = {"images": len(records), "apples": count}
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: apple\n",
        encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print("YOLO configuration:", out / "data.yaml")

if __name__ == "__main__":
    main()
