from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Predict on an image
results = model.predict(
    source="test_images/handknife.jpg",
    conf=0.25,
    save=True
)

print("Detection Completed!")
