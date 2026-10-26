"""Fine-tune an apple-only YOLO detector."""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    a = p.parse_args()
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, device=a.device, project="runs",
                name="apple_detector", exist_ok=True,
                hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
                scale=0.5, fliplr=0.5,
                degrees=8, translate=0.1,
                mosaic=0.8, close_mosaic=10, patience=20)
    print("Best weights: runs/apple_detector/weights/best.pt")

if __name__ == "__main__":
    main()
