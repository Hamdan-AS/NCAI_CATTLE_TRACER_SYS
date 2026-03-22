import streamlit as st
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import tempfile
import sqlite3
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Cattle Tag Detector", layout="wide")

DB_PATH = "cattle_logs.db"

# =========================
# DATABASE
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            tag TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def insert_log(tag, conf):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO logs (timestamp, tag, confidence) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), tag, conf)
    )
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM logs ORDER BY id DESC", conn)
    conn.close()
    return df

# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_model(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        return YOLO(tmp.name)

# =========================
# DETECTION
# =========================
def process_frame(frame, model, conf_thres):
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        detections.append((label, conf))
        insert_log(label, conf)

    return frame, detections

# =========================
# IMAGE PROCESSING
# =========================
def process_image(file, model, conf):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    frame, dets = process_frame(frame, model, conf)
    return frame, dets

# =========================
# VIDEO PROCESSING
# =========================
def process_video(file, model, conf):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    cap = cv2.VideoCapture(tfile.name)

    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count > 100:
            break

        if count % 10 == 0:
            frame, _ = process_frame(frame, model, conf)
            frames.append(frame)

        count += 1

    cap.release()
    return frames

# =========================
# UI
# =========================
def main():
    init_db()

    st.title("🐄 Cattle Collar Tag Detection (YOLOv8)")

    # Sidebar
    st.sidebar.header("Settings")

    model_file = st.sidebar.file_uploader("Upload YOLO Model (.pt)", type=["pt"])

    conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

    if model_file:
        model = load_model(model_file)
        st.success("Model Loaded!")

        tab1, tab2, tab3 = st.tabs(["Image", "Video", "Analytics"])

        # IMAGE TAB
        with tab1:
            img = st.file_uploader("Upload Image", type=["jpg", "png"])
            if img:
                result, dets = process_image(img, model, conf)
                st.image(result, channels="BGR")

                if dets:
                    st.write("Detections:")
                    st.write(dets)

        # VIDEO TAB
        with tab2:
            vid = st.file_uploader("Upload Video", type=["mp4", "avi"])
            if vid:
                if st.button("Process Video"):
                    frames = process_video(vid, model, conf)

                    for f in frames:
                        st.image(f, channels="BGR")

        # ANALYTICS TAB
        with tab3:
            df = get_logs()
            st.dataframe(df)

            if not df.empty:
                st.bar_chart(df["confidence"])

    else:
        st.warning("Upload a YOLO model to start.")

# =========================
if __name__ == "__main__":
    main()
