from alert_system import trigger_alert
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/best.pt")

# Ask user for video path
video_path = input("Enter video path: ")

# Predict on video
results = model.predict(
    source=video_path,
    conf=0.25,
    save=True
)

# Process detections
for r in results:
    annotated_frame = r.plot()

    for box in r.boxes:
        confidence = float(box.conf[0])
        cls = int(box.cls[0])
        label = r.names[cls]

        if confidence > 0.5:
            annotated_frame = trigger_alert(
                annotated_frame,
                label,
                confidence,
                box
            )

print("Video Detection Completed!")