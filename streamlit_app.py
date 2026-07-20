import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import glob
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("🛡️ Weapon Detection System")
st.markdown("### AI-Based CCTV Surveillance using YOLOv8")

st.success("System Status : Ready")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Choose Detection Mode",
    (
        "📷 Image Detection",
        "🎥 Video Detection",
        "📹 Webcam Detection"
    )
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Project

Weapon Detection using YOLOv8

### Team Members

• Khushi

• Runjhan

• Nisha
"""
)

# ======================================================
# IMAGE DETECTION
# ======================================================

if mode == "📷 Image Detection":

    st.header("📷 Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image,use_container_width=True)

        if st.button("🔍 Detect Weapon"):

            with st.spinner("Detecting..."):

                # Save Image Temporarily
                with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as temp:

                    image.save(temp.name)

                    temp_path = temp.name

                # Prediction
                results = model.predict(
                    source=temp_path,
                    conf=0.25
                )

                os.remove(temp_path)

            if len(results)>0:

                detected_image = results[0].plot()

                with col2:

                    st.subheader("Detection Result")

                    st.image(
                        detected_image,
                        channels="BGR",
                        use_container_width=True
                    )

                if len(results[0].boxes)>0:

                    st.success("✅ Weapon Detected")

                    st.markdown("### Detection Details")

                    for box in results[0].boxes:

                        cls = int(box.cls[0])

                        conf = float(box.conf[0])

                        st.write(f"**Class :** {results[0].names[cls]}")

                        st.progress(conf)

                        st.write(f"Confidence : **{conf:.2f}**")

                else:

                    st.info("No Weapon Detected.")

# ======================================================
# VIDEO DETECTION
# ======================================================

elif mode == "🎥 Video Detection":

    st.header("🎥 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("🎥 Detect Weapon"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                temp.write(uploaded_video.getbuffer())
                input_video = temp.name

            output_video = "output_detected.mp4"

            cap = cv2.VideoCapture(input_video)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0:
                fps = 25

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_video,
                fourcc,
                fps,
                (width, height)
            )

            progress = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_no = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(
                frame,
                conf=0.25,
                verbose=False
                )

                annotated = results[0].plot()

                out.write(annotated)

                frame_no += 1

                if total_frames > 0:
                    progress.progress(frame_no / total_frames)

            cap.release()
            out.release()
            if not os.path.exists(output_video):
                st.error("Output video was not created.")
                st.stop()

            if os.path.getsize(output_video) == 0:
                st.error("Output video is empty.")
                st.stop()

            st.success("✅ Detection Completed!")

            st.subheader("Detected Video")

            # File save hone ka wait
            out.release()
            cap.release()

            import time
            time.sleep(1)

            st.success("✅ Detection Completed!")

            st.subheader("Detected Video")

            with open(output_video, "rb") as f:
                video_bytes = f.read()

            col1, col2, col3 = st.columns([1,2,1])

            with col2:
                st.video(video_bytes, format="video/mp4")

            st.download_button(
                label="📥 Download Detected Video",
                data=video_bytes,
                file_name="output_detected.mp4",
                mime="video/mp4"
            )

            os.remove(input_video)

# ======================================================
# WEBCAM DETECTION
# ======================================================

elif mode == "📹 Webcam Detection":

    st.header("📹 Live Webcam Detection")

    if st.button("▶ Start Webcam"):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Unable to open webcam.")
            st.stop()

        # Streamlit placeholders
        status = st.empty()
        details = st.empty()

        st.info("Press 'Q' on the webcam window to stop.")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO Prediction
            results = model.predict(
                frame,
                conf=0.25,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # Detection Status
            if len(results[0].boxes) > 0:

                detected_objects = []

                for box in results[0].boxes:

                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = results[0].names[cls]

                    detected_objects.append(
                        f"{label} ({confidence:.2f})"
                    )

                # Detection Status
                if len(results[0].boxes) > 0:

                    detected_objects = []

                    weapon_found = False

                    for box in results[0].boxes:

                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = results[0].names[cls]

                        # Person ko ignore karo
                        if label.lower() == "person":
                            continue

                        weapon_found = True

                        detected_objects.append(
                            f"🔫 {label} ({confidence:.2f})"
                        )

                    if weapon_found:

                        status.error("🚨 WEAPON DETECTED")

                        details.markdown(
                            "### Detection Details\n\n"
                            + "\n".join(detected_objects)
                        )

                    else:

                        status.info("🟢 Only Person Detected")

                        details.markdown("""
                ### Detection Details

                No weapon detected.
                """)

                else:

                    status.info("🟢 No Detection")

                    details.empty()

            else:

                status.info("🟢 No Weapon Detected")

                details.markdown("""
### 🔍 Detection Details

**Class:** None

**Confidence:** 0.00
""")

            # Show Webcam
            cv2.imshow(
                "Weapon Detection - Webcam",
                annotated_frame
            )

            # Press Q to Stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        status.warning("📴 Webcam Closed")