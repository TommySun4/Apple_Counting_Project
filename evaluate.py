"""Compute detection mAP and image-level counting error on labeled test split."""
import argparse
import csv
import math
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/apple_detector/weights/best.pt")
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    a = p.parse_args()
    model = YOLO(a.weights)
    metrics = model.val(data=a.data, split="test", imgsz=a.imgsz, max_det=1000)
    images = sorted(p for p in (Path(a.dataset) / "images" / "test").iterdir()
                    if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"})
    if not images:
        raise RuntimeError("Test split is empty")
    rows = []
    for path in images:
        label = Path(a.dataset) / "labels" / "test" / f"{path.stem}.txt"
        true_count = len([l for l in label.read_text().splitlines() if l.strip()]) if label.exists() else 0
        pred = model.predict(str(path), conf=a.conf, imgsz=a.imgsz,
                             iou=0.5, max_det=1000, verbose=False)[0]
        pred_count = len(pred.boxes) if pred.boxes is not None else 0
        rows.append((path.name, true_count, pred_count, abs(true_count-pred_count)))
    mae = sum(r[3] for r in rows)/len(rows)
    rmse = math.sqrt(sum((r[1]-r[2])**2 for r in rows)/len(rows))
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    with (out / "test_errors.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "predicted", "absolute_error"])
        w.writerows(rows)
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"Count MAE: {mae:.3f}  Count RMSE: {rmse:.3f}")
    print("Per-image results: outputs/test_errors.csv")

if __name__ == "__main__":
    main()
