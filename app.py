import os

while True:
    print("\n========== Weapon Detection System ==========")
    print("1. Image Detection")
    print("2. Video Detection")
    print("3. Webcam Detection")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        os.system("python image_detection.py")

    elif choice == "2":
        os.system("python video_detection.py")

    elif choice == "3":
        os.system("python webcam_detection.py")

    elif choice == "4":
        print("Exiting...")
        break

    else:
        print("Invalid Choice!")