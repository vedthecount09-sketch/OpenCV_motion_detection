import cv2
import streamlit as st
import numpy as np
import os
import time

st.title("Motion Detection and Marking")

# Use this video file from your MDM folder
DEFAULT_VIDEO_PATH = r"D:\Vedang\COEP\T.Y\Sem 6\MDM\horse_10sec.mp4"

mode = st.radio("Select input source:", ("Video File", "Upload Video", "Webcam"))
frame_placeholder = st.empty()
min_area = st.slider("Minimum contour area for motion", min_value=1, max_value=1000, value=20, step=1)

# Handle video upload
uploaded_video = None
if mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        st.success(f"Video uploaded: {uploaded_video.name}")
    else:
        st.info("Please upload a video file to start motion detection.")

if "run_video" not in st.session_state:
    st.session_state.run_video = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        st.session_state.run_video = True
with col2:
    if st.button("Stop"):
        st.session_state.run_video = False

cap = None
if st.session_state.run_video:
    if mode == "Video File":
        if os.path.exists(DEFAULT_VIDEO_PATH):
            st.write(f"Using video: `{DEFAULT_VIDEO_PATH}`")
            cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
        else:
            st.error(f"Video file not found: {DEFAULT_VIDEO_PATH}")
    elif mode == "Upload Video":
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            temp_video_path = f"temp_{uploaded_video.name}"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            cap = cv2.VideoCapture(temp_video_path)
        else:
            st.error("Please upload a video file first.")
    else:
        cap = cv2.VideoCapture(0)

    if cap is not None:
        ret, prev_frame = cap.read()
        if not ret:
            st.error("Could not read video or webcam.")
            cap.release()
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            while cap.isOpened() and st.session_state.run_video:
                ret, current_frame = cap.read()
                if not ret:
                    break

                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, current_gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                frame_placeholder.image(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                prev_gray = current_gray
                time.sleep(0.03)

            cap.release()
            st.write("Video finished or stopped.")
            
            # Clean up temporary file if it exists
            if mode == "Upload Video" and uploaded_video is not None:
                temp_video_path = f"temp_{uploaded_video.name}"
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
