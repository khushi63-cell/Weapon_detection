from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train-7/weights/best.pt")

# Run prediction on an image
results = model("dataset/weapon_dataset/test/images/armas-1435-_jpg.rf.04a5a3c592b98cdb3ab1e42b9f4258de.jpg", save=True)

print("Detection Completed!")