import cv2
from ultralytics import YOLO
from pathlib import Path

# Get project root directory
ROOT = Path(__file__).resolve().parent.parent

# Load trained model
model = YOLO(ROOT / "runs" / "detect" / "train-7" / "weights" / "best.pt")
print("Classes:", model.names)

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam Started. Press 'q' to quit.")

while True:

    # Capture frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Run YOLO detection with confidence threshold
    results = model(frame, conf=0.50)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Variable to check if weapon is detected
    weapon_detected = False

    # Check every detected object
    for box in results[0].boxes:

        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])

        # Print every detection (optional)
        print(f"{name}: {conf:.2f}")

        # If weapon detected
        if name.lower() in ["knife", "weapon", "gun", "pistol"]:

            weapon_detected = True

            print("\n========================")
            print("⚠ WEAPON DETECTED!")
            print(f"Type       : {name}")
            print(f"Confidence : {conf:.2f}")
            print("========================\n")

    # Display warning on screen
    if weapon_detected:
        cv2.putText(
            annotated_frame,
            "WEAPON DETECTED!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),   # Red color (BGR)
            3
        )

    # Show webcam
    cv2.imshow("Weapon Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()