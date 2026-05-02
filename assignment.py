import cv2
import streamlit as st
import numpy as np
import os
import time

st.title("Motion Detection and Marking")
st.write("Upload a video file or use your webcam for motion detection.")

mode = st.radio("Select input source:", ("Upload Video", "Webcam"))
frame_placeholder = st.empty()
min_area = st.slider("Minimum contour area for motion", min_value=1, max_value=1000, value=20, step=1)

if "run_video" not in st.session_state:
    st.session_state.run_video = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        st.session_state.run_video = True
with col2:
    if st.button("Stop"):
        st.session_state.run_video = False

uploaded_video = None
temp_video_path = None
if mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        extension = os.path.splitext(uploaded_video.name)[1]
        temp_video_path = f"temp_uploaded_video{extension}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.success(f"Uploaded video: {uploaded_video.name}")
    else:
        st.info("Please upload a video file to start motion detection.")

cap = None
if st.session_state.run_video:
    if mode == "Upload Video":
        if uploaded_video is not None and temp_video_path is not None:
            cap = cv2.VideoCapture(temp_video_path)
        else:
            st.error("Please upload a video file first.")
    else:
        cap = cv2.VideoCapture(0)

    if cap is not None and cap.isOpened():
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
    elif cap is not None:
        st.error("Could not open video source.")

if temp_video_path and os.path.exists(temp_video_path):
    try:
        os.remove(temp_video_path)
    except OSError:
        pass
