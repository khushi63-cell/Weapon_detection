import cv2
from ultralytics import YOLO

# 1. Load the trained YOLOv8 model (best.pt)
model = YOLO('best.pt')

# 2. Initialize the webcam or CCTV camera (0 is the default laptop webcam)
cap = cv2.VideoCapture("sample.jpg")

print("Weapon Detection System Started... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # 3. Perform prediction on the current frame
    results = model(frame)

    # 4. Process detections and trigger alerts
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = r.names[cls]

            # Trigger alert only if detection confidence is above 50%
            if conf > 0.5:
                # Print alert message to the console
                print(f"⚠️ ALERT: {label.upper()} DETECTED! Confidence: {conf:.2f}")

                # Draw a Red Bounding Box around the detected weapon
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Display the alert text on the video frame
                cv2.putText(frame, f"ALERT: {label.upper()}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 5. Display the output video feed
    cv2.imshow('CCTV Weapon Detection Alert System', frame)

    # Press 'q' on the keyboard to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all active windows
cap.release()
cv2.destroyAllWindows()