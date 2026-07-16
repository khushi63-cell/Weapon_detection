from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Detect objects in video
results = model.predict(
    source="test_videos/knife.mp4",
    conf=0.25,
    save=True
)

print("Video Detection Completed!")