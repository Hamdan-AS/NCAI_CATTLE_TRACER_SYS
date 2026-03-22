"""
CattleEye — Cattle Behaviour & Collar Tag Monitoring System
============================================================
Single-file Streamlit application.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §1  IMPORTS & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

import io
import logging
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DB_PATH: str = "cattle_analytics.db" [cite: 9]
_db_lock = threading.Lock()   # Ensures thread-safe SQLite writes [cite: 9]

BEHAVIOUR_LABELS: dict[int, str] = {
    0: "Grazing",
    1: "Resting",
    2: "Walking",
    3: "Running",
    4: "Drinking",
    5: "Unknown",
} [cite: 10]

BBOX_COLOUR     = (0,   200, 100)
TAG_BBOX_COLOUR = (255, 165,   0)
TEXT_COLOUR     = (255, 255, 255)
CV_FONT         = cv2.FONT_HERSHEY_SIMPLEX [cite: 11]
FONT_SCALE      = 0.6
THICKNESS       = 2


# ═══════════════════════════════════════════════════════════════════════════════
# §3  DATABASE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: str = DB_PATH) -> None:
    """Initialize SQLite database and logs table[cite: 12]."""
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        TEXT    NOT NULL,
                    tag_id           TEXT    NOT NULL,
                    behavior_label   TEXT    NOT NULL,
                    confidence_score REAL    NOT NULL
                )
                """
            ) [cite: 12, 13, 14]
            conn.commit()
        finally:
            conn.close()

def bulk_insert_detections(records: list[dict], db_path: str = DB_PATH) -> None:
    """Efficiently insert multiple detection records[cite: 17]."""
    if not records:
        return
    timestamp = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    rows = [
        (timestamp, r["tag_id"], r["behavior_label"], round(float(r["confidence_score"]), 4))
        for r in records
    ] [cite: 17]
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.executemany(
                "INSERT INTO logs (timestamp, tag_id, behavior_label, confidence_score) VALUES (?, ?, ?, ?)",
                rows,
            ) [cite: 18]
            conn.commit()
        finally:
            conn.close() [cite: 19]

def fetch_all_logs(db_path: str = DB_PATH) -> pd.DataFrame:
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            df = pd.read_sql_query("SELECT * FROM logs ORDER BY id DESC", conn)
        finally:
            conn.close()
    return df

def fetch_unique_tag_count(db_path: str = DB_PATH) -> int:
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT tag_id) FROM logs")
            result = cursor.fetchone()
        finally:
            conn.close()
    return result[0] if result else 0 [cite: 20, 21]

def fetch_behavior_summary(db_path: str = DB_PATH) -> pd.DataFrame:
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            df = pd.read_sql_query(
                "SELECT behavior_label, COUNT(*) AS count FROM logs GROUP BY behavior_label ORDER BY count DESC",
                conn,
            ) [cite: 22]
        finally:
            conn.close()
    return df

def fetch_recent_logs(limit: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            df = pd.read_sql_query("SELECT * FROM logs ORDER BY id DESC LIMIT ?", conn, params=(limit,)) [cite: 23]
        finally:
            conn.close()
    return df

def clear_all_logs(db_path: str = DB_PATH) -> None:
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute("DELETE FROM logs") [cite: 24]
            conn.commit()
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# §4  INFERENCE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_bytes: bytes, model_name: str):
    """Load a YOLO model from raw bytes[cite: 25, 26, 27]."""
    try:
        from ultralytics import YOLO
        tmp_path = Path(f"/tmp/{model_name}")
        tmp_path.write_bytes(model_bytes)
        model = YOLO(str(tmp_path))
        logger.info("Model '%s' loaded successfully.", model_name) [cite: 28]
        return model
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", model_name, exc)
        st.error(f"❌ Could not load model: {exc}") [cite: 29]
        return None

def _get_ocr_reader():
    """Lazy-initialise EasyOCR reader[cite: 30]."""
    if "ocr_reader" not in st.session_state:
        try:
            import easyocr
            st.session_state["ocr_reader"] = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialised.") [cite: 31]
        except Exception as exc:
            logger.warning("EasyOCR unavailable: %s", exc)
            st.session_state["ocr_reader"] = None
    return st.session_state["ocr_reader"]

def run_ocr(cropped_image: np.ndarray) -> str:
    """Run OCR on cropped BGR image[cite: 32]."""
    if cropped_image is None or cropped_image.size == 0:
        return "UNKNOWN"
    reader = _get_ocr_reader()
    if reader is None:
        return "UNKNOWN"
    try:
        rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb, detail=0, paragraph=False)
        if not results:
            return "UNKNOWN" [cite: 33]
        raw = " ".join(results)
        cleaned = re.sub(r"[^A-Z0-9\-]", "", raw.upper()).strip()
        return cleaned if cleaned else "UNKNOWN"
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return "UNKNOWN"

def _resolve_behaviour(class_id: int) -> str:
    return BEHAVIOUR_LABELS.get(int(class_id), "Unknown") [cite: 34]

def _draw_annotation(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, colour: tuple) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, CV_FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), CV_FONT, FONT_SCALE, TEXT_COLOUR, THICKNESS) [cite: 35]

def process_frame(frame: np.ndarray, model, confidence_threshold: float = 0.40) -> tuple:
    """Run YOLO inference on a single frame[cite: 36]."""
    if frame is None or frame.size == 0:
        return frame, []
    detections = []
    try:
        results = model(frame, verbose=False)[0]
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return frame, [] [cite: 37]
    annotated = frame.copy()
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[max(0, y1):y2, max(0, x1):x2] [cite: 38]
        tag_id = run_ocr(crop)
        behaviour = _resolve_behaviour(cls_id)
        label = f"{tag_id} | {behaviour} {conf:.0%}" [cite: 39]
        colour = TAG_BBOX_COLOUR if tag_id != "UNKNOWN" else BBOX_COLOUR
        _draw_annotation(annotated, x1, y1, x2, y2, label, colour)
        detections.append({"tag_id": tag_id, "behavior_label": behaviour, "confidence_score": conf})
    return annotated, detections

def process_image(image_bytes: bytes, model, confidence_threshold: float = 0.40) -> tuple:
    """Decode and process an uploaded image[cite: 40]."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image file.")
        return process_frame(frame, model, confidence_threshold)
    except Exception as exc:
        logger.error("Image processing error: %s", exc) [cite: 41]
        st.error(f"❌ Image processing failed: {exc}")
        return None, []

def process_video(
    video_bytes: bytes,
    model,
    confidence_threshold: float = 0.40,
    frame_skip: int = 5,
    max_frames: int = 300,
    progress_callback=None,
) -> tuple:
    """
    Decodes and processes video frames. 
    FIXED: Removed the 'break' that stopped processing after one instance[cite: 48].
    """
    annotated_frames = []
    all_detections = []
    tmp_path = Path("/tmp/_uploaded_video.mp4")
    try:
        tmp_path.write_bytes(video_bytes)
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened(): [cite: 45]
            st.error("❌ Cannot open video format.")
            return annotated_frames, all_detections
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_index = 0
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret: [cite: 46]
                break
            if frame_index % frame_skip == 0:
                annotated, detections = process_frame(frame, model, confidence_threshold)
                annotated_frames.append(annotated)
                all_detections.extend(detections)
                processed += 1 [cite: 47]
                if progress_callback:
                    progress_callback(min(frame_index / total_frames, 1.0))
                if processed >= max_frames:
                    break # Safety limit reached
            frame_index += 1
    finally:
        cap.release()
    return annotated_frames, all_detections

def frame_to_rgb(bgr_frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) [cite: 49]

def frames_to_gif_bytes(frames: list[np.ndarray], fps: int = 10) -> bytes:
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(frame_to_rgb(f)) for f in frames]
        buf = io.BytesIO()
        pil_frames[0].save(buf, format="GIF", save_all=True, append_images=pil_frames[1:], loop=0, duration=int(1000/fps)) [cite: 50]
        return buf.getvalue()
    except Exception:
        return b""


# ═══════════════════════════════════════════════════════════════════════════════
# §5  PAGE CONFIG & UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="CattleEye", page_icon="🐄", layout="wide", initial_sidebar_state="expanded") [cite: 51]

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; } [cite: 52]
      :root { --bg: #0f1117; --surface: #1a1d27; --accent: #e8a838; --text: #e4e6ed; } [cite: 53]
      .stApp { background: var(--bg); color: var(--text); } [cite: 55]
      [data-testid="stMetric"] { background: var(--surface); border-radius: 8px; padding: 12px; } [cite: 57, 58]
      .tag-badge { background: var(--accent); color: #0f1117; font-family: 'IBM Plex Mono'; padding: 2px 8px; border-radius: 4px; } [cite: 69, 70]
    </style>
    """,
    unsafe_allow_html=True,
)

def _init_session_state() -> None:
    defaults = {
        "model": None, "model_name": None, "session_tag_ids": set(),
        "video_frames": [], "confidence_threshold": 0.40, "frame_skip": 5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value [cite: 72]


# ═══════════════════════════════════════════════════════════════════════════════
# §6  UI RENDERING (SIDEBAR & TABS)
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🐄 CattleEye") [cite: 74]
        model_file = st.file_uploader("Upload YOLO Weights", type=["pt", "onnx"])
        if model_file:
            if st.session_state["model_name"] != model_file.name:
                with st.spinner("Loading..."):
                    model = load_model(model_file.read(), model_file.name) [cite: 75, 76]
                    if model:
                        st.session_state["model"] = model
                        st.session_state["model_name"] = model_file.name
                        st.success(f"Loaded {model_file.name}") [cite: 77]
        st.divider()
        st.session_state["confidence_threshold"] = st.slider("Confidence", 0.1, 0.95, st.session_state["confidence_threshold"]) [cite: 79]
        st.session_state["frame_skip"] = st.slider("Frame Skip", 1, 30, st.session_state["frame_skip"])
        st.metric("Unique Tags (DB)", fetch_unique_tag_count()) [cite: 80]
        st.metric("Total Logs", len(fetch_all_logs())) [cite: 80]
        if st.button("🗑️ Clear Logs"):
            clear_all_logs()
            st.rerun() [cite: 84]

def render_image_tab():
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png"], key="img_up") [cite: 87]
    if uploaded and st.session_state["model"]:
        with st.spinner("Processing..."):
            annotated, detections = process_image(uploaded.read(), st.session_state["model"], st.session_state["confidence_threshold"]) [cite: 88]
        if annotated is not None:
            bulk_insert_detections(detections) [cite: 85]
            st.image(frame_to_rgb(annotated), use_container_width=True) [cite: 89]
            st.dataframe(pd.DataFrame(detections), use_container_width=True)

def render_video_tab():
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi"], key="vid_up") [cite: 91]
    if uploaded and st.session_state["model"]:
        if st.button("▶️ Process Video"):
            with st.spinner("Analyzing Every Sampled Instance..."):
                frames, detections = process_video(uploaded.read(), st.session_state["model"], st.session_state["confidence_threshold"], st.session_state["frame_skip"]) [cite: 93]
            if frames:
                st.session_state["video_frames"] = frames
                bulk_insert_detections(detections)
                st.success(f"Processed {len(frames)} frames.") [cite: 94]
        if st.session_state["video_frames"]:
            idx = st.slider("Frame Browser", 0, len(st.session_state["video_frames"])-1) [cite: 95]
            st.image(frame_to_rgb(st.session_state["video_frames"][idx]), use_container_width=True)

def render_analytics_tab():
    df = fetch_all_logs() [cite: 100]
    if not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Behaviour")
            fig = px.bar(fetch_behavior_summary(), x="behavior_label", y="count") [cite: 101, 102]
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.head(100), use_container_width=True) [cite: 104]

def main():
    _init_session_state()
    init_db()
    st.title("🐄 CattleEye Monitoring") [cite: 106, 107]
    render_sidebar()
    t1, t2, t3 = st.tabs(["🖼️ Image", "🎥 Video", "📈 Analytics"])
    with t1: render_image_tab()
    with t2: render_video_tab() [cite: 109]
    with t3: render_analytics_tab()

if __name__ == "__main__":
    main()
