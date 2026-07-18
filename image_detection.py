from alert_system import trigger_alert
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/best.pt")

# Ask user for image path
image_path = input("Enter image path: ")

# Predict on image
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# Check detections
for r in results:
    annotated_image = r.plot()   # Image with YOLO bounding boxes

    for box in r.boxes:
        confidence = float(box.conf[0])
        cls = int(box.cls[0])
        label = r.names[cls]

        if confidence > 0.5:
            annotated_image = trigger_alert(
                annotated_image,
                label,
                confidence,
                box
            )

# Show final image
cv2.imshow("Weapon Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detection Completed!")
