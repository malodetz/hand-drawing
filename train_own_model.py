from ultralytics import YOLO


model = YOLO("yolo11n-pose.pt")
results = model.train(
    data="hand-keypoints.yaml",
    epochs=100,
    imgsz=224,
    batch=16,
    device=0,
    verbose=True,
    save=True,
    patience=5,
)
