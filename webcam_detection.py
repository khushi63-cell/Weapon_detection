import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model.predict(frame, conf=0.25)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Weapon Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()