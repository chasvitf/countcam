"""Export YOLO11n to the ONNX file the web demo loads (public/models/yolo11n.onnx)."""
import shutil
from pathlib import Path

from ultralytics import YOLO

out = YOLO("yolo11n.pt").export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
target = Path(__file__).resolve().parent.parent / "public" / "models" / "yolo11n.onnx"
shutil.move(out, target)
print(f"saved {target}")
