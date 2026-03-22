"""
CattleEye — Cattle Behaviour & Collar Tag Monitoring System
============================================================
Single-file Streamlit application.

All logic is organised into clearly labelled sections:

  §1  Imports & Logging
  §2  Constants & Configuration
  §3  Database Layer          (SQLite CRUD, thread-safe)
  §4  Inference Layer         (YOLO model loading, OCR, frame processing)
  §5  Streamlit Page Config   (theme CSS, session-state init)
  §6  Sidebar                 (model upload, settings, live stats)
  §7  Tab — Image Inference
  §8  Tab — Video Inference
  §9  Tab — Analytics & Export
  §10 Main Entrypoint

Run:
    streamlit run app.py
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

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH: str = "cattle_analytics.db"
_db_lock = threading.Lock()   # Ensures thread-safe SQLite writes

# ── Behaviour label mapping ───────────────────────────────────────────────────
# Extend / reorder to match your custom YOLO model's class indices.
BEHAVIOUR_LABELS: dict[int, str] = {
    0: "Grazing",
    1: "Resting",
    2: "Walking",
    3: "Running",
    4: "Drinking",
    5: "Unknown",
}

# ── Bounding-box visual settings ──────────────────────────────────────────────
BBOX_COLOUR     = (0,   200, 100)   # Green  – generic cattle detection
TAG_BBOX_COLOUR = (255, 165,   0)   # Orange – collar tag identified via OCR
TEXT_COLOUR     = (255, 255, 255)   # White  – label text
CV_FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.6
THICKNESS       = 2


# ═══════════════════════════════════════════════════════════════════════════════
# §3  DATABASE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: str = DB_PATH) -> None:
    """
    Create the SQLite database and `logs` table if they do not yet exist.
    Safe to call on every app start — uses CREATE TABLE IF NOT EXISTS.
    """
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
            )
            conn.commit()
        finally:
            conn.close()


def insert_detection(
    tag_id: str,
    behavior_label: str,
    confidence_score: float,
    db_path: str = DB_PATH,
) -> None:
    """Insert a single detection event into the logs table."""
    timestamp = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute(
                "INSERT INTO logs (timestamp, tag_id, behavior_label, confidence_score) "
                "VALUES (?, ?, ?, ?)",
                (timestamp, tag_id, behavior_label, round(float(confidence_score), 4)),
            )
            conn.commit()
        finally:
            conn.close()


def bulk_insert_detections(
    records: list[dict],
    db_path: str = DB_PATH,
) -> None:
    """
    Efficiently insert multiple detection records in a single transaction.
    Each record dict must contain: 'tag_id', 'behavior_label', 'confidence_score'.
    """
    if not records:
        return
    timestamp = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    rows = [
        (timestamp, r["tag_id"], r["behavior_label"], round(float(r["confidence_score"]), 4))
        for r in records
    ]
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.executemany(
                "INSERT INTO logs (timestamp, tag_id, behavior_label, confidence_score) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()


def fetch_all_logs(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return all log entries as a DataFrame, newest first."""
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            df = pd.read_sql_query("SELECT * FROM logs ORDER BY id DESC", conn)
        finally:
            conn.close()
    return df


def fetch_unique_tag_count(db_path: str = DB_PATH) -> int:
    """Return the number of distinct tag IDs in the database."""
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT tag_id) FROM logs")
            result = cursor.fetchone()
        finally:
            conn.close()
    return result[0] if result else 0


def fetch_behavior_summary(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return detection counts grouped by behaviour label."""
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            df = pd.read_sql_query(
                "SELECT behavior_label, COUNT(*) AS count "
                "FROM logs GROUP BY behavior_label ORDER BY count DESC",
                conn,
            )
        finally:
            conn.close()
    return df


def fetch_recent_logs(limit: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    """Return the most recent N log entries."""
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM logs ORDER BY id DESC LIMIT ?",
                conn, params=(limit,),
            )
        finally:
            conn.close()
    return df


def clear_all_logs(db_path: str = DB_PATH) -> None:
    """Delete all records from the logs table."""
    with _db_lock:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute("DELETE FROM logs")
            conn.commit()
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# §4  INFERENCE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4a  Model Loading ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_bytes: bytes, model_name: str):
    """
    Load a YOLO model from raw bytes.
    Cached globally by (model_bytes, model_name) so the model is instantiated
    only once per session regardless of Streamlit reruns.

    Returns the Ultralytics YOLO model instance, or None on failure.
    """
    try:
        from ultralytics import YOLO   # Deferred import — faster cold start

        tmp_path = Path(f"/tmp/{model_name}")
        tmp_path.write_bytes(model_bytes)
        model = YOLO(str(tmp_path))
        logger.info("Model '%s' loaded successfully.", model_name)
        return model
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", model_name, exc)
        st.error(f"❌ Could not load model: {exc}")
        return None


# ── 4b  OCR Engine ────────────────────────────────────────────────────────────

def _get_ocr_reader():
    """
    Lazy-initialise EasyOCR and store in st.session_state so it survives
    reruns without re-downloading language models on every interaction.
    Returns an easyocr.Reader, or None if EasyOCR is unavailable.
    """
    if "ocr_reader" not in st.session_state:
        try:
            import easyocr  # noqa: PLC0415

            st.session_state["ocr_reader"] = easyocr.Reader(
                ["en"], gpu=False, verbose=False
            )
            logger.info("EasyOCR reader initialised.")
        except Exception as exc:
            logger.warning("EasyOCR unavailable — OCR disabled: %s", exc)
            st.session_state["ocr_reader"] = None
    return st.session_state["ocr_reader"]


def run_ocr(cropped_image: np.ndarray) -> str:
    """
    Run OCR on a cropped BGR image and return a clean alphanumeric tag ID
    string (e.g. 'UK-0042').  Returns 'UNKNOWN' on any failure.
    """
    if cropped_image is None or cropped_image.size == 0:
        return "UNKNOWN"

    reader = _get_ocr_reader()
    if reader is None:
        return "UNKNOWN"

    try:
        rgb     = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb, detail=0, paragraph=False)
        if not results:
            return "UNKNOWN"
        raw     = " ".join(results)
        # Keep only letters, digits, and hyphens — standard ear-tag format
        cleaned = re.sub(r"[^A-Z0-9\-]", "", raw.upper()).strip()
        return cleaned if cleaned else "UNKNOWN"
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return "UNKNOWN"


# ── 4c  Detection & Annotation Helpers ───────────────────────────────────────

def _resolve_behaviour(class_id: int) -> str:
    """Map a YOLO class index to a human-readable behaviour label."""
    return BEHAVIOUR_LABELS.get(int(class_id), "Unknown")


def _draw_annotation(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    colour: tuple[int, int, int],
) -> None:
    """Draw a labelled, filled-header bounding box on *frame* in-place."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, CV_FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), CV_FONT, FONT_SCALE, TEXT_COLOUR, THICKNESS)


# ── 4d  Frame Processing ──────────────────────────────────────────────────────

def process_frame(
    frame: np.ndarray,
    model,
    confidence_threshold: float = 0.40,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run YOLO inference on a single BGR frame.

    Returns:
        (annotated_frame, detections)
        where each detection dict has keys:
          'tag_id', 'behavior_label', 'confidence_score'
    """
    if frame is None or frame.size == 0:
        return frame, []

    detections: list[dict] = []
    try:
        results = model(frame, verbose=False)[0]
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return frame, []

    annotated = frame.copy()

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        cls_id        = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop      = frame[max(0, y1):y2, max(0, x1):x2]
        tag_id    = run_ocr(crop)
        behaviour = _resolve_behaviour(cls_id)
        label     = f"{tag_id} | {behaviour} {conf:.0%}"
        colour    = TAG_BBOX_COLOUR if tag_id != "UNKNOWN" else BBOX_COLOUR

        _draw_annotation(annotated, x1, y1, x2, y2, label, colour)
        detections.append(
            {"tag_id": tag_id, "behavior_label": behaviour, "confidence_score": conf}
        )

    return annotated, detections


# ── 4e  Image Pipeline ────────────────────────────────────────────────────────

def process_image(
    image_bytes: bytes,
    model,
    confidence_threshold: float = 0.40,
) -> tuple[Optional[np.ndarray], list[dict]]:
    """Decode an uploaded image, run detection, return (annotated_frame, detections)."""
    try:
        arr   = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None — file may be corrupt.")
        return process_frame(frame, model, confidence_threshold)
    except Exception as exc:
        logger.error("Image processing error: %s", exc)
        st.error(f"❌ Image processing failed: {exc}")
        return None, []


# ── 4f  Video Pipeline ────────────────────────────────────────────────────────

def process_video(
    video_bytes: bytes,
    model,
    confidence_threshold: float = 0.40,
    frame_skip: int = 5,
    max_frames: int = 300,
    progress_callback=None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Decode an uploaded video, sample every N-th frame, run detection, and
    return (annotated_frames, all_detections).

    Args:
        frame_skip:        Process every N-th frame (performance tuning).
        max_frames:        Hard cap on sampled frames to prevent OOM.
        progress_callback: Optional callable(float) for UI progress bar updates.
    """
    annotated_frames: list[np.ndarray] = []
    all_detections:   list[dict]       = []

    tmp_path = Path("/tmp/_uploaded_video.mp4")
    try:
        tmp_path.write_bytes(video_bytes)
    except Exception as exc:
        logger.error("Cannot write video to disk: %s", exc)
        st.error(f"❌ Video save failed: {exc}")
        return annotated_frames, all_detections

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        st.error("❌ Cannot open video — file may be corrupt or in an unsupported format.")
        return annotated_frames, all_detections

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_index  = 0
    processed    = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                annotated, detections = process_frame(frame, model, confidence_threshold)
                annotated_frames.append(annotated)
                all_detections.extend(detections)
                processed += 1

                if progress_callback:
                    progress_callback(min(frame_index / total_frames, 1.0))

                if processed >= max_frames:
                    logger.info("Max frame limit (%d) reached.", max_frames)
                    break

            frame_index += 1
    finally:
        cap.release()

    logger.info(
        "Video processed: %d total frames, %d sampled, %d detections.",
        frame_index, processed, len(all_detections),
    )
    return annotated_frames, all_detections


# ── 4g  Conversion Helpers ────────────────────────────────────────────────────

def frame_to_rgb(bgr_frame: np.ndarray) -> np.ndarray:
    """Convert a BGR numpy frame to RGB for Streamlit's st.image()."""
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)


def frames_to_gif_bytes(frames: list[np.ndarray], fps: int = 10) -> bytes:
    """Encode a list of BGR frames into an animated GIF and return raw bytes."""
    try:
        from PIL import Image  # noqa: PLC0415

        pil_frames = [Image.fromarray(frame_to_rgb(f)) for f in frames]
        buf        = io.BytesIO()
        pil_frames[0].save(
            buf, format="GIF", save_all=True,
            append_images=pil_frames[1:], loop=0,
            duration=int(1000 / fps),
        )
        return buf.getvalue()
    except Exception as exc:
        logger.warning("GIF export failed: %s", exc)
        return b""


# ═══════════════════════════════════════════════════════════════════════════════
# §5  STREAMLIT PAGE CONFIG, CSS & SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CattleEye – Monitoring System",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');

      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

      :root {
          --bg:      #0f1117;
          --surface: #1a1d27;
          --border:  #2b2f3e;
          --accent:  #e8a838;
          --accent2: #4caf77;
          --text:    #e4e6ed;
          --muted:   #7b8194;
          --danger:  #e05252;
      }

      .stApp { background: var(--bg); color: var(--text); }

      section[data-testid="stSidebar"] {
          background: var(--surface);
          border-right: 1px solid var(--border);
      }

      [data-testid="stMetric"] {
          background: var(--surface);
          border: 1px solid var(--border);
          border-left: 3px solid var(--accent);
          border-radius: 8px;
          padding: 12px 16px;
      }
      [data-testid="stMetricLabel"] { color: var(--muted) !important; }
      [data-testid="stMetricValue"] {
          color: var(--accent) !important;
          font-family: 'IBM Plex Mono', monospace;
      }

      h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
      h1 { color: var(--accent); font-size: 1.7rem; }
      h2 { color: var(--text); font-size: 1.2rem;
           border-bottom: 1px solid var(--border); padding-bottom: 6px; }

      [data-testid="stFileUploader"] {
          background: var(--surface);
          border: 1px dashed var(--border);
          border-radius: 8px;
          padding: 8px;
      }

      .stButton > button {
          background: var(--accent); color: #0f1117;
          font-weight: 700; border: none; border-radius: 6px;
          padding: 8px 20px; transition: opacity 0.2s;
      }
      .stButton > button:hover { opacity: 0.85; }

      [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

      .tag-badge {
          display: inline-block; background: var(--accent); color: #0f1117;
          font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
          font-weight: 700; padding: 2px 8px; border-radius: 4px; margin: 2px;
      }
      .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    """Populate st.session_state with safe defaults on first run."""
    defaults = {
        "model":                None,
        "model_name":           None,
        "session_tag_ids":      set(),
        "last_annotated_frame": None,
        "last_detections":      [],
        "video_frames":         [],
        "confidence_threshold": 0.40,
        "frame_skip":           5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ═══════════════════════════════════════════════════════════════════════════════
# §6  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Model upload, inference settings, and live statistics panel."""
    with st.sidebar:
        st.markdown("## 🐄 CattleEye")
        st.markdown(
            "<span style='font-size:0.8rem;color:#7b8194;'>"
            "Behaviour & Collar Tag Monitor</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Model Upload ──────────────────────────────────────────────────
        st.markdown("### ⚙️ Model Weights")
        model_file = st.file_uploader(
            "Upload `.pt` or `.onnx`",
            type=["pt", "onnx"],
            key="model_uploader",
            help="YOLOv8 / YOLOv9 / YOLOv10 weights supported.",
        )

        if model_file is not None:
            if st.session_state["model_name"] != model_file.name:
                with st.spinner("Loading model…"):
                    model = load_model(
                        model_bytes=model_file.read(),
                        model_name=model_file.name,
                    )
                if model is not None:
                    st.session_state["model"]      = model
                    st.session_state["model_name"] = model_file.name
                    st.success(f"✅ Loaded: **{model_file.name}**")
                else:
                    st.error("❌ Model load failed.")
            else:
                st.info(f"✅ Active: **{st.session_state['model_name']}**")
        else:
            st.caption("No model loaded — upload weights above to enable detection.")

        st.divider()

        # ── Inference Settings ────────────────────────────────────────────
        st.markdown("### 🎛️ Inference Settings")
        st.session_state["confidence_threshold"] = st.slider(
            "Confidence Threshold",
            min_value=0.10, max_value=0.95,
            value=st.session_state["confidence_threshold"],
            step=0.05, format="%.2f",
        )
        st.session_state["frame_skip"] = st.slider(
            "Video Frame Skip  (process every N-th frame)",
            min_value=1, max_value=30,
            value=st.session_state["frame_skip"], step=1,
        )

        st.divider()

        # ── Live Statistics ───────────────────────────────────────────────
        st.markdown("### 📊 Live Statistics")
        st.metric("Unique Tag IDs (DB)", fetch_unique_tag_count())
        st.metric("Total Detections (DB)", len(fetch_all_logs()))

        beh_df = fetch_behavior_summary()
        if not beh_df.empty:
            st.markdown("**Behaviour Breakdown**")
            for _, row in beh_df.iterrows():
                st.markdown(
                    f"<span class='tag-badge'>{row['behavior_label']}</span> {row['count']}",
                    unsafe_allow_html=True,
                )

        if st.session_state["session_tag_ids"]:
            st.markdown("**Tags this Session**")
            st.markdown(
                "".join(
                    f"<span class='tag-badge'>{t}</span>"
                    for t in sorted(st.session_state["session_tag_ids"])
                ),
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Danger Zone ───────────────────────────────────────────────────
        with st.expander("⚠️ Danger Zone"):
            if st.button("🗑️ Clear All Logs", type="secondary"):
                clear_all_logs()
                st.session_state["session_tag_ids"] = set()
                st.success("All logs cleared.")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# §7  TAB — IMAGE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _store_detections(detections: list[dict]) -> None:
    """Persist detections to DB and update the in-session tag ID set."""
    if detections:
        bulk_insert_detections(detections)
        for det in detections:
            if det["tag_id"] != "UNKNOWN":
                st.session_state["session_tag_ids"].add(det["tag_id"])


def _detection_summary(detections: list[dict]) -> None:
    """Render a 3-column KPI row summarising a list of detections."""
    if not detections:
        st.info("No detections above the confidence threshold.")
        return

    unique_tags = {d["tag_id"] for d in detections if d["tag_id"] != "UNKNOWN"}
    avg_conf    = sum(d["confidence_score"] for d in detections) / len(detections)

    c1, c2, c3 = st.columns(3)
    c1.metric("Detections",     len(detections))
    c2.metric("Unique Tags",    len(unique_tags))
    c3.metric("Avg Confidence", f"{avg_conf:.1%}")

    if unique_tags:
        st.markdown(
            "**Tags identified:** " + "".join(
                f"<span class='tag-badge'>{t}</span>" for t in sorted(unique_tags)
            ),
            unsafe_allow_html=True,
        )


def render_image_tab() -> None:
    st.markdown("## 🖼️ Image Inference")

    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader"
    )
    if uploaded is None:
        st.info("Upload a `.jpg` or `.png` to begin cattle detection.")
        return
    if st.session_state["model"] is None:
        st.warning("⚠️ No model loaded — upload model weights in the sidebar first.")
        return

    with st.spinner("Running inference…"):
        annotated, detections = process_image(
            image_bytes=uploaded.read(),
            model=st.session_state["model"],
            confidence_threshold=st.session_state["confidence_threshold"],
        )

    if annotated is None:
        st.error("❌ Failed to process image.")
        return

    _store_detections(detections)

    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.image(frame_to_rgb(annotated), caption="Annotated Detection",
                 use_container_width=True)
    with col_info:
        st.markdown("### 🔍 Detection Results")
        _detection_summary(detections)
        if detections:
            df = pd.DataFrame(detections)
            df["confidence_score"] = df["confidence_score"].map("{:.2%}".format)
            st.dataframe(df, use_container_width=True, hide_index=True)

    _, enc = cv2.imencode(".jpg", annotated)
    st.download_button(
        "⬇️ Download Annotated Image",
        data=enc.tobytes(),
        file_name=f"annotated_{uploaded.name}",
        mime="image/jpeg",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §8  TAB — VIDEO INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def render_video_tab() -> None:
    st.markdown("## 🎥 Video Inference")

    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "avi"], key="video_uploader"
    )
    if uploaded is None:
        st.info("Upload a `.mp4` or `.avi` to begin frame-by-frame detection.")
        return
    if st.session_state["model"] is None:
        st.warning("⚠️ No model loaded — upload model weights in the sidebar first.")
        return

    col_run, _ = st.columns([1, 3])
    run_btn    = col_run.button("▶️ Process Video", use_container_width=True)

    if run_btn:
        video_bytes  = uploaded.read()
        progress_bar = st.progress(0.0, text="Processing frames…")
        status_text  = st.empty()

        def _on_progress(p: float) -> None:
            progress_bar.progress(p, text=f"Processing… {p:.0%}")

        with st.spinner("Extracting and analysing frames…"):
            frames, detections = process_video(
                video_bytes=video_bytes,
                model=st.session_state["model"],
                confidence_threshold=st.session_state["confidence_threshold"],
                frame_skip=st.session_state["frame_skip"],
                progress_callback=_on_progress,
            )

        progress_bar.empty()

        if not frames:
            st.error("❌ No frames could be extracted from the video.")
            return

        st.session_state["video_frames"] = frames
        _store_detections(detections)
        status_text.success(
            f"✅ Processed {len(frames)} frames — {len(detections)} total detections."
        )

        st.markdown("### 🔍 Detection Results")
        _detection_summary(detections)

        st.markdown("### 🎞️ Frame Browser")
        frame_idx = st.slider(
            "Browse annotated frames",
            min_value=0, max_value=max(len(frames) - 1, 0),
            step=1, key="frame_slider",
        )
        st.image(
            frame_to_rgb(frames[frame_idx]),
            caption=f"Frame {frame_idx + 1} / {len(frames)}",
            use_container_width=True,
        )

        if len(frames) > 1:
            gif_bytes = frames_to_gif_bytes(frames[:60], fps=8)
            if gif_bytes:
                st.download_button(
                    "⬇️ Download Preview GIF (first 60 frames)",
                    data=gif_bytes, file_name="cattle_preview.gif", mime="image/gif",
                )

        if detections:
            with st.expander("📋 All Detections (this video)"):
                df = pd.DataFrame(detections)
                df["confidence_score"] = df["confidence_score"].map("{:.2%}".format)
                st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  TAB — ANALYTICS & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _plotly_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    colour_scale: list[str],
    labels: dict,
) -> None:
    """Render a dark-themed Plotly bar chart with transparent background."""
    fig = px.bar(
        df, x=x, y=y, color=y,
        color_continuous_scale=colour_scale, labels=labels,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e4e6ed", coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="#2b2f3e"),
        yaxis=dict(gridcolor="#2b2f3e"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_analytics_tab() -> None:
    st.markdown("## 📈 Analytics & Data Export")

    col_refresh, col_export, _ = st.columns([1, 1, 3])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    df_all = fetch_all_logs()
    if df_all.empty:
        st.info("No data yet — run some detections to populate the log.")
        return

    # ── KPI Row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Logs",     len(df_all))
    k2.metric("Unique Tags",    df_all["tag_id"].nunique())
    k3.metric("Avg Confidence", f"{df_all['confidence_score'].mean():.1%}")
    k4.metric("Top Behaviour",  df_all["behavior_label"].mode()[0])

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Behaviour Distribution")
        beh_df = fetch_behavior_summary()
        if not beh_df.empty:
            _plotly_bar(
                beh_df, x="behavior_label", y="count",
                colour_scale=["#2b2f3e", "#e8a838"],
                labels={"behavior_label": "Behaviour", "count": "Detections"},
            )

    with col_b:
        st.markdown("### Tag Frequency (Top 10)")
        tag_freq = (
            df_all[df_all["tag_id"] != "UNKNOWN"]
            .groupby("tag_id").size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        if not tag_freq.empty:
            _plotly_bar(
                tag_freq, x="tag_id", y="count",
                colour_scale=["#2b2f3e", "#4caf77"],
                labels={"tag_id": "Tag ID", "count": "Detections"},
            )
        else:
            st.info("No identified tags yet.")

    st.divider()

    # ── Recent Logs Table ─────────────────────────────────────────────────
    st.markdown("### 🗃️ Recent Logs (latest 100)")
    recent = fetch_recent_logs(limit=100)
    if not recent.empty:
        display = recent.copy()
        display["confidence_score"] = display["confidence_score"].map("{:.2%}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── CSV Export ────────────────────────────────────────────────────────
    with col_export:
        st.download_button(
            "⬇️ Export Full DB (.csv)",
            data=df_all.to_csv(index=False).encode("utf-8"),
            file_name="cattle_analytics_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §10  MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session_state()
    init_db()

    st.markdown(
        """
        <div style='margin-bottom:8px;'>
            <h1>🐄 CattleEye</h1>
            <p style='color:#7b8194; margin-top:-10px; font-size:0.9rem;'>
                Behaviour Classification &amp; Collar Tag Monitoring System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    tab_img, tab_vid, tab_ana = st.tabs(
        ["🖼️  Image Inference", "🎥  Video Inference", "📈  Analytics & Export"]
    )
    with tab_img:
        render_image_tab()
    with tab_vid:
        render_video_tab()
    with tab_ana:
        render_analytics_tab()


if __name__ == "__main__":
    main()
