import cv2

def trigger_alert(frame, label, confidence, box):
    print(f"⚠ ALERT: {label.upper()} DETECTED! Confidence: {confidence:.2f}")

    x1, y1 = map(int, box.xyxy[0][:2])

    cv2.putText(
        frame,
        f"ALERT: {label.upper()}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
    )

    return frame