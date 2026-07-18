from alert_system import trigger_alert
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'Q' to quit the webcam.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model.predict(frame, conf=0.25)

    for r in results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = r.names[cls]

            # Trigger alert if confidence > 50%
            if confidence > 0.5:
                frame = trigger_alert(frame, label, confidence, box)

    # Show the modified frame
    cv2.imshow("Weapon Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Closing webcam...")
        break

cap.release()
cv2.destroyAllWindows()