import cv2
from ultralytics import YOLO
from pathlib import Path

# Project folder
ROOT = Path(__file__).resolve().parent.parent

# Load trained model
model = YOLO(ROOT / "runs" / "detect" / "train-7" / "weights" / "best.pt")

# Load video
video_path = ROOT / "videos" / "test.mp4"
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Cannot open video")
    exit()

print("Video Started...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Weapon Detection Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Video Completed")