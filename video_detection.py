from ultralytics import YOLO
from alert_system import trigger_alert
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# -----------------------------
# Load Model
# -----------------------------
model = YOLO("models/best.pt")

# -----------------------------
# Select Video
# -----------------------------
root = Tk()
root.withdraw()

video_path = askopenfilename(
    title="Select Video",
    filetypes=[
        ("Video Files", "*.mp4 *.avi *.mov"),
        ("All Files", "*.*")
    ]
)

if not video_path:
    print("No video selected.")
    exit()

# -----------------------------
# Open Video
# -----------------------------
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

# -----------------------------
# Output Video
# -----------------------------
output_path = "output_detected.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("Processing...")

# -----------------------------
# Frame-by-frame Detection
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25)

    annotated = results[0].plot()

    if len(results[0].boxes) > 0:

        for box in results[0].boxes:

            confidence = float(box.conf[0])

            cls = int(box.cls[0])

            label = results[0].names[cls]

            if confidence >= 0.50:

                annotated = trigger_alert(
                    annotated,
                    label,
                    confidence,
                    box
                )

    writer.write(annotated)

cap.release()
writer.release()

print("Done!")
print("Saved as:", os.path.abspath(output_path))