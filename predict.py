"""Count apples in an image or folder and save annotated images and count.csv."""
import argparse
import csv
from pathlib import Path
import cv2
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def image_paths(path):
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTS)
    raise FileNotFoundError(path)

def run(model_path, source, output, confidence=0.25, imgsz=960, clahe=False):
    model = YOLO(model_path)
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for f in image_paths(source):
        image = cv2.imread(str(f))
        if image is None:
            print(f"Skipping unreadable image: {f}")
            continue
        inp = image
        if clahe:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            inp = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        result = model.predict(inp, conf=confidence, imgsz=imgsz,
                               iou=0.5, max_det=1000, verbose=False)[0]
        boxes = result.boxes
        n = len(boxes) if boxes is not None else 0
        annotated = image.copy()
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                score = float(box.conf[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 210, 50), 2)
                cv2.putText(annotated, f"apple {score:.2f}",
                            (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (50, 210, 50), 2)
        cv2.rectangle(annotated, (0, 0), (min(image.shape[1]-1, 330), 45), (20, 40, 55), -1)
        cv2.putText(annotated, f"Apple Count: {n}", (12, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        relative = f.relative_to(source) if source.is_dir() else Path(f.name)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), annotated):
            raise RuntimeError(f"Failed to write {target}")
        csv_rows.append((str(relative), n))
        print(f"{relative}: {n} apples")
    with (output / "counts.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "predicted_count"])
        writer.writerows(csv_rows)
    print(f"Saved {len(csv_rows)} images to {output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/apple_detector/weights/best.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="outputs")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--clahe", action="store_true")
    a = p.parse_args()
    run(a.weights, a.source, a.output, a.conf, a.imgsz, a.clahe)
