from ultralytics import YOLO
model = YOLO("/Users/agraw1/Desktop/Fall2025/Intro-to-parallel/project/Code/yolo_finetuned_bdd100k.pt")  # or your finetuned .pt

model.export(
    format="onnx",
    opset=12,          # ✨ REQUIRED for OpenCV
    simplify=True,
    dynamic=False       # recommended
)