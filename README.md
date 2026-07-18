# Weapon Detection System using YOLOv8

## Project Overview

The Weapon Detection System is a Computer Vision and Deep Learning based application developed to detect weapons such as knives using the YOLOv8 object detection model. The system can detect weapons from images, videos, and live webcam feeds in real-time and display alerts whenever a weapon is detected.

This project was developed as a group academic project using Python, OpenCV, and Ultralytics YOLOv8.

## Features

-  Weapon Detection from Images
-  Weapon Detection from Videos
-  Live Webcam Detection
-  Real-time Alert System
-  YOLOv8 Custom Trained Model
-  Automatic Result Saving
-  Modular Project Structure


## 🛠 Technologies Used

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- Git & GitHub
- VS Code

## Project Structure

```
WeaponDetection/
│
├── models/
│   └── best.pt
│
├── test_images/
├── test_videos/
│
├── image_detection.py
├── video_detection.py
├── webcam_detection.py
├── alert_system.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/khushi63-cell/Weapon_detection.git
```

### Move into project folder

```bash
cd Weapon_detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the application:

```bash
python app.py
```

Choose one of the following options:

1. Image Detection
2. Video Detection
3. Webcam Detection
4. Exit

---

## Image Detection

- Enter the image path.
- The system detects weapons.
- Detection result is saved automatically.

---

## Video Detection

- Enter the video path.
- YOLO processes the complete video.
- Output video is saved automatically.

---

## Webcam Detection

- Opens the webcam.
- Performs real-time weapon detection.
- Displays alert messages on the screen.
- Press **Q** to close the webcam.

---

## Alert System

Whenever a weapon is detected:

- Alert message is displayed.
- Detection confidence is shown.
- Bounding boxes are drawn around detected objects.

---

## Model Information

Model Used:

- YOLOv8 (Ultralytics)

Custom Model:

- best.pt

Confidence Threshold:

```
0.25
```

Alert Threshold:

```
0.50
```

## Future Improvements

- Graphical User Interface (Tkinter/PyQt)
- Email Alert System
- SMS Notification
- Cloud Database Integration
- CCTV Camera Integration
- Multi-Weapon Detection
- Face Recognition Integration

---

## Team Members
- Khushi (03501042025)
- Runjhan (05601042025)
- Kumari Nisha (03701042025)

## License

This project is developed for educational and academic purposes only.

## Acknowledgement

We would like to thank our project guide and faculty members for their continuous guidance and support throughout the development of this project.
