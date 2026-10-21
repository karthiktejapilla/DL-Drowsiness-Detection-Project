"""
Human Drowsiness Detection — Deep Learning Web Application.

A modern, professional computer vision dashboard for driver drowsiness detection.
Features:
- Shared eye-state inference pipeline with MediaPipe attempt and OpenCV Haar fallback
- Unified inference engine across Image Upload, Video Analysis, and Live Webcam
- Dual-eye localization, bilateral state resolution, and confidence scoring
- Video temporal tracking with rolling PERCLOS, blink counter, and CSV export
- Browser camera snapshot analysis and native OpenCV live monitor guide
- Comprehensive Project Overview, Architecture, Dataset Explorer, and Evaluation Benchmarks
"""
import json
import os
import sys
import tempfile
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset_inspector import DatasetInspector
from src.drowsiness_engine import DrowsinessEngine
from src.inference import ImageAnalysisResult, InferenceEngine
from src.model import load_trained_model


# ─── Page Configuration ───
st.set_page_config(
    page_title="Human Drowsiness Detection — AI Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Global Custom Styling (Modern Dark Navy & Cyan Theme) ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Backgrounds & Text */
    .stApp {
        background: radial-gradient(circle at 50% -10%, #0d1a2d 0%, #080f1a 50%, #050912 100%);
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #080e18 !important;
        border-right: 1px solid #1a273a;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(13, 26, 45, 0.9) 0%, rgba(16, 37, 66, 0.8) 50%, rgba(8, 15, 26, 0.95) 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4, #3b82f6, #6366f1);
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 960px;
        line-height: 1.65;
    }

    /* Custom Cards */
    .custom-card {
        background-color: #0e1726;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .custom-card:hover {
        border-color: #0ea5e9;
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(14, 165, 233, 0.15);
    }
    .card-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .card-desc {
        color: #94a3b8;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* Badges & Status */
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .badge-awake {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .badge-closed {
        background-color: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }
    .badge-drowsy {
        background-color: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border: 1px solid #ef4444;
        animation: pulse 1.8s infinite;
    }
    .badge-warn {
        background-color: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.45);
    }
    .badge-info {
        background-color: rgba(56, 189, 248, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.45);
    }

    /* Formula & Code Boxes */
    .formula-box {
        background-color: #080d16;
        border-left: 4px solid #0ea5e9;
        padding: 12px 18px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        margin: 12px 0;
        color: #e2e8f0;
    }

    /* Telemetry Pill */
    .telemetry-chip {
        display: inline-flex;
        align-items: center;
        background: #111e33;
        border: 1px solid #1f3554;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* Notice & Disclaimers */
    .disclaimer-card {
        background-color: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(234, 179, 8, 0.35);
        border-left: 5px solid #eab308;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 20px 0;
        font-size: 0.89rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Buttons & Inputs */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #1e3a5f;
        background: #0f233d;
        color: #f8fafc;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #06b6d4;
        background: #0891b2;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Model & Inference Pipeline ───
@st.cache_resource
def get_cached_inference_engine():
    """Load model once and retain in memory across Streamlit reruns."""
    if not os.path.exists(config.EYE_STATE_MODEL_PATH):
        return None
    try:
        model = load_trained_model(config.EYE_STATE_MODEL_PATH)
        return InferenceEngine(model=model)
    except Exception as e:
        st.error(f"Failed to initialize inference pipeline: {e}")
        return None


# ─── Sidebar Navigation ───
def render_sidebar(engine: Optional[InferenceEngine]) -> str:
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 12px 0 18px 0;">
        <span style="font-size: 2.5rem;">🚗</span>
        <h2 style="margin: 6px 0 2px 0; font-size: 1.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">Drowsiness AI</h2>
        <p style="margin: 0; font-size: 0.82rem; color: #0ea5e9; font-weight: 600;">Computer Vision Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Engine Status Widget
    if engine is not None:
        st.sidebar.markdown("""
        <div style="background-color: #0b1322; border: 1px solid #1a2a44; border-radius: 10px; padding: 12px 14px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 6px;">
                <span style="color: #94a3b8;">Neural Backbone:</span>
                <span class="status-badge badge-awake">ACTIVE</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 6px;">
                <span style="color: #94a3b8;">Vision Pipeline:</span>
                <span style="color: #38bdf8; font-weight: 600;">FaceMesh / Haar</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem;">
                <span style="color: #94a3b8;">Execution Target:</span>
                <span style="color: #a78bfa; font-weight: 600;">Optimized CPU</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("⚠️ Model not found at models/eye_state_model.keras")

    st.sidebar.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #64748b; margin-bottom: 8px; text-transform: uppercase;'>Navigation</p>", unsafe_allow_html=True)
    pages = [
        "🏠 Home / Dashboard",
        "🖼️ Image Analysis",
        "🎬 Video Analysis",
        "📹 Live Webcam",
        "📖 Project Overview",
        "🏛️ System Architecture",
        "📊 Model Training & Evaluation",
        "🔍 Dataset Explorer",
        "ℹ️ About / Limitations",
    ]
    selected_page = st.sidebar.radio("Navigate to:", pages, index=0, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.76rem; color: #64748b; line-height: 1.5; padding: 0 4px;">
        <strong>Human Drowsiness Detection</strong><br>
        Framework: TensorFlow 2.17 + OpenCV<br>
        Models: MobileNetV3 + Dual-Ocular Tracker<br>
        Datasets: MRL Eye + UTA-RLDD
    </div>
    """, unsafe_allow_html=True)

    return selected_page


# ─── PAGE 1: Home / Dashboard ───
def render_home_page():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Human Drowsiness Detection</div>
        <div class="hero-subtitle">
            An experimental computer vision project that classifies eye crops as open or closed, then tracks eye closure over time in video.
            It combines <strong>OpenCV face and eye cascades</strong>, a <strong>MobileNetV3Small eye classifier</strong>,
            and a <strong>temporal decision engine</strong> for prolonged closure indicators.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Action Cards
    st.markdown("<h3 style='margin-bottom: 14px;'>🚀 Analysis Workspaces</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🖼️ Single Image Analysis</div>
            <div class="card-desc">
                Analyze a portrait or eye crop. FaceMesh is attempted when its legacy API is available; OpenCV Haar face and eye cascades provide the fallback.
                Manual eye-region cropping is available if automatic detection misses.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🎬 Video Sequence Analysis</div>
            <div class="card-desc">
                Process driving video recordings (.mp4, .avi, .mov). Calculates frame-by-frame temporal closure, blink statistics,
                rolling PERCLOS metrics, timestamped drowsiness event logs, and downloadable telemetry CSVs.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">📹 Live Camera Monitoring</div>
            <div class="card-desc">
                Capture browser snapshots for instant ocular diagnosis or launch the dedicated local continuous OpenCV webcam stream
                featuring heads-up telemetry overlays and native acoustic warning alerts.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Real System Status Metrics
    st.markdown("<h3 style='margin-top: 24px; margin-bottom: 14px;'>⚡ Verified Platform Status</h3>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    has_model = os.path.isfile(config.EYE_STATE_MODEL_PATH)
    m1.metric("Neural Checkpoint", "Ready (9.5 MB)" if has_model else "Missing")
    m2.metric("Face & Eye Localization", "Haar fallback")
    m3.metric("Local MRL Dataset", "84,898 Images (37 Subj)")
    m4.metric("UTA-RLDD Dataset", "11,787 Video Frames")

    # Workflow Visualization
    st.markdown("<h3 style='margin-top: 28px; margin-bottom: 14px;'>🔬 Overall System Workflow</h3>", unsafe_allow_html=True)
    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #38bdf8; font-weight: 700; margin-bottom: 8px;">1. Image / Video / Camera Frame</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                Uploaded stills and video frames use the same inference engine. The browser camera supplies snapshots; the local webcam script supplies continuous frames.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #818cf8; font-weight: 700; margin-bottom: 8px;">2. Face and Eye Crops</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                The engine tries MediaPipe FaceMesh when available, then OpenCV Haar cascades. If automatic eye detection misses, image analysis offers a manual ROI crop.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #34d399; font-weight: 700; margin-bottom: 8px;">3. Eye-State Classification</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                Each crop is converted to grayscale, resized to 128×128, scaled to [-1, 1], then classified as Awake (0) or Sleepy / closed (1).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step4:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #f59e0b; font-weight: 700; margin-bottom: 8px;">4. Temporal Decision Engine</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                Video and continuous webcam predictions feed the temporal engine for closure duration, blink counts, alerts, and PERCLOS. A still image cannot establish drowsiness duration.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ Non-Diagnostic Engineering Prototype Notice:</strong>
        This application is an experimental deep learning computer vision system. Facial and ocular metrics serve as indicators
        of eye closure and visual fatigue, not certified medical diagnoses of clinical drowsiness or autonomous vehicular safety compliance.
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE 2: Single Image Analysis ───
def render_image_page(engine: Optional[InferenceEngine]):
    st.markdown("<h2 style='margin-bottom: 4px;'>🖼️ Single Image Ocular Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 22px;'>Analyze driver portraits, webcam stills, or standalone eye crops with multi-tier detection and bilateral telemetry.</p>", unsafe_allow_html=True)

    if engine is None:
        st.error("Inference Engine unavailable. Please verify model checkpoint.")
        return

    # Quick Samples Bar
    st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 8px;'>🧪 Load Bundled Test Sample:</p>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    sample_path = None

    if b1.button("👁️ Awake Eye (MRL)", use_container_width=True):
        sample_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_eye_awake.png")
    if b2.button("😴 Sleepy Eye (MRL)", use_container_width=True):
        sample_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_eye_sleepy.png")
    if b3.button("👤 Active Driver Face", use_container_width=True):
        sample_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_face_active.jpg")
    if b4.button("🥱 Fatigue Driver Face", use_container_width=True):
        sample_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_face_fatigue.jpg")

    # Upload & Controls Row
    col_up, col_ctrl = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Image File (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            help="Upload a driver photograph, snapshot, or cropped eye region."
        )

    with col_ctrl:
        st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 4px;'>⚙️ Processing Strategy:</p>", unsafe_allow_html=True)
        analysis_mode = st.radio(
            "Detection Mode",
            ["🤖 Auto-Detect (Face & Eyes)", "👁️ Force Direct Eye Crop", "✂️ Manual ROI Selection"],
            index=0,
            label_visibility="collapsed",
        )

    image_to_process = None

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Image exceeds the 10 MB upload limit.")
            return
        file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_to_process is None:
            st.error("Failed to decode uploaded image. Ensure valid JPG or PNG format.")
            return
    elif sample_path and os.path.isfile(sample_path):
        image_to_process = cv2.imread(sample_path)
        st.success(f"Loaded bundled asset: `{os.path.basename(sample_path)}`")

    if image_to_process is not None:
        ih, iw = image_to_process.shape[:2]

        # Manual ROI parameters if selected
        manual_box = None
        if analysis_mode == "✂️ Manual ROI Selection":
            with st.expander("✂️ Interactive Region-of-Interest (ROI) Selector", expanded=True):
                st.caption("Select the precise ocular region to analyze. Adjust percentage coordinates below:")
                preset = st.selectbox("Quick Region Preset", ["Custom Coordinates", "Left Eye Area", "Right Eye Area", "Upper Face Center"])
                if preset == "Left Eye Area":
                    d_x, d_y, d_w, d_h = 18, 24, 32, 28
                elif preset == "Right Eye Area":
                    d_x, d_y, d_w, d_h = 52, 24, 32, 28
                elif preset == "Upper Face Center":
                    d_x, d_y, d_w, d_h = 20, 20, 60, 35
                else:
                    d_x, d_y, d_w, d_h = 25, 25, 50, 40

                c_x, c_y = st.columns(2)
                p_x = c_x.slider("Left Boundary (%)", 0, 90, d_x)
                p_y = c_y.slider("Top Boundary (%)", 0, 90, d_y)
                c_w, c_h = st.columns(2)
                p_w = c_w.slider("Width (%)", 5, 100 - p_x, d_w)
                p_h = c_h.slider("Height (%)", 5, 100 - p_y, d_h)

                rx1 = int(iw * p_x / 100.0)
                ry1 = int(ih * p_y / 100.0)
                rx2 = min(iw, rx1 + int(iw * p_w / 100.0))
                ry2 = min(ih, ry1 + int(ih * p_h / 100.0))
                manual_box = (rx1, ry1, rx2, ry2)

                # Show cropped ROI preview
                roi_preview = image_to_process[ry1:ry2, rx1:rx2]
                if roi_preview.size > 0:
                    st.image(cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB), caption=f"Selected Eye ROI ({rx2-rx1}×{ry2-ry1} px)", width=220)

        # Run unified inference
        is_direct = (analysis_mode == "👁️ Force Direct Eye Crop")
        res: ImageAnalysisResult = engine.process_frame(
            image_to_process,
            allow_direct_crop_fallback=(analysis_mode != "🤖 Auto-Detect (Face & Eyes)"),
            manual_roi=manual_box,
            is_direct_crop_hint=is_direct,
            draw_annotations=True,
        )

        # Side-by-side Visual Display
        st.markdown("---")
        col_in, col_out = st.columns(2)

        with col_in:
            st.markdown("#### 📷 Input Frame")
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"""
            <div style="margin-top: 6px;">
                <span class="telemetry-chip">📐 {iw}×{ih} px</span>
                <span class="telemetry-chip">🎨 BGR 3-Channel</span>
                <span class="telemetry-chip">Mode: {analysis_mode}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_out:
            st.markdown("#### 🧠 Neural Detection Overlay")
            st.image(cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Determine badge styling
            if res.overall_class == 0:
                badge_class = "badge-awake"
            elif res.overall_class == 1:
                badge_class = "badge-closed"
            else:
                badge_class = "badge-warn"

            st.markdown(f"""
            <div style="margin-top: 8px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600;">Ocular Status:</span>
                    <span class="status-badge {badge_class}" style="margin-left: 8px; font-size: 0.92rem;">{res.overall_state}</span>
                </div>
                <span class="telemetry-chip">Method: {res.detection_method}</span>
            </div>
            """, unsafe_allow_html=True)

        # Telemetry KPI Row
        st.markdown("<h4 style='margin-top: 24px;'>📊 Quantitative Ocular Telemetry</h4>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Prediction", res.overall_state)
        k2.metric("Neural Confidence", f"{res.overall_confidence:.1%}")
        k3.metric("Detected Eye Regions", f"{res.eyes_detected_count} Regions")
        k4.metric("Inference Latency", f"{res.latency_ms:.1f} ms")

        # Bilateral Eye Breakdown Cards
        if res.left_eye or res.right_eye:
            st.markdown("#### 👁️ Bilateral Ocular Breakdown")
            eye_c1, eye_c2 = st.columns(2)

            for eye_res, col in [(res.left_eye, eye_c1), (res.right_eye, eye_c2)]:
                if eye_res is not None:
                    with col:
                        eye_badge = "badge-awake" if eye_res.pred_class == 0 else "badge-closed"
                        st.markdown(f"""
                        <div class="custom-card">
                            <div class="card-title">
                                <span>{eye_res.name}</span>
                                <span class="status-badge {eye_badge}">{eye_res.pred_label}</span>
                            </div>
                        """, unsafe_allow_html=True)

                        col_sub_img, col_sub_data = st.columns([1, 2])
                        with col_sub_img:
                            if eye_res.crop_bgr.size > 0:
                                st.image(cv2.cvtColor(eye_res.crop_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with col_sub_data:
                            st.write(f"• **Confidence:** `{eye_res.confidence:.1%}`")
                            st.write(f"• **Raw Sigmoid:** `{eye_res.raw_probability:.3f}`")
                            st.write(f"• **Coordinates:** `{eye_res.bbox}`")
                        st.markdown("</div>", unsafe_allow_html=True)

        if not res.face_detected and res.eyes_detected_count == 0:
            st.warning("⚠️ " + res.message)


# ─── PAGE 3: Video Sequence Analysis ───
def render_video_page(engine: Optional[InferenceEngine]):
    st.markdown("<h2 style='margin-bottom: 4px;'>🎬 Video Sequence Drowsiness Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 22px;'>Continuous temporal evaluation calculating frame-by-frame closure duration, blink counts, and rolling PERCLOS.</p>", unsafe_allow_html=True)

    if engine is None:
        st.error("Inference Engine unavailable. Please verify model checkpoint.")
        return

    # Configuration Parameters
    with st.expander("⚙️ Temporal Analysis Parameters & Sensitivity", expanded=False):
        c1, c2, c3 = st.columns(3)
        sample_step = c1.slider("Sampling Step (Process every Nth frame)", 1, 6, int(config.VIDEO_FRAME_SAMPLE_RATE), help="Optimizes CPU processing throughput.")
        closure_sec = c2.slider("Drowsiness Alert Threshold (seconds)", 1.0, 4.0, float(config.EYE_CLOSURE_THRESHOLD_SEC), 0.2, help="Continuous closure duration triggering an alert.")
        max_frames = c3.number_input("Maximum Sampled Frames", 100, 10000, 1500, 100)

    # Video Source
    use_bundled = st.checkbox("Use Bundled Driver Video (sample_data/sample_driver_clip.mp4)", value=False)
    video_source_path = None
    uploaded_vid = None

    if use_bundled:
        bundled_p = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_driver_clip.mp4")
        if os.path.isfile(bundled_p):
            video_source_path = bundled_p
            st.info("Loaded bundled test video: `sample_driver_clip.mp4`")
    else:
        uploaded_vid = st.file_uploader("Upload Driving Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    decode_path = None
    temp_file_to_clean = None

    if uploaded_vid is not None:
        video_bytes = uploaded_vid.getvalue()
        if len(video_bytes) > 200 * 1024 * 1024:
            st.error("Video exceeds the 200 MB upload limit.")
            return
        t_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1])
        t_file.write(video_bytes)
        t_file.close()
        decode_path = t_file.name
        temp_file_to_clean = t_file.name
        st.video(video_bytes)
    elif video_source_path:
        decode_path = video_source_path

    if decode_path:
        cap = cv2.VideoCapture(decode_path)
        if not cap.isOpened():
            cap.release()
            if temp_file_to_clean and os.path.exists(temp_file_to_clean):
                os.unlink(temp_file_to_clean)
            st.error("Failed to decode video container. Check video codec.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_s = total_frames / fps if fps > 0 else 0.0
        w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.markdown(f"""
        <div style="background-color: #0b1322; border: 1px solid #1a2a44; border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
            <span class="telemetry-chip">📐 {w_v}×{h_v} px</span>
            <span class="telemetry-chip">🎞️ Total Frames: {total_frames}</span>
            <span class="telemetry-chip">⏱️ Duration: {duration_s:.1f}s</span>
            <span class="telemetry-chip">🚀 Native FPS: {fps:.1f}</span>
            <span class="telemetry-chip">⚡ Sample Rate: Every {sample_step} frames</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Start Temporal Video Analysis", type="primary"):
            temporal_engine = DrowsinessEngine(closure_threshold_sec=closure_sec)
            cap = cv2.VideoCapture(decode_path)

            col_disp, col_kpi = st.columns([3, 2])
            with col_disp:
                video_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

            with col_kpi:
                st.markdown("#### ⚡ Live Video Telemetry")
                kpi_state = st.empty()
                kpi_closure = st.empty()
                kpi_blinks = st.empty()
                kpi_perclos = st.empty()
                kpi_alert = st.empty()

            frame_records = []
            alert_events = []
            raw_idx = 0
            analyzed_idx = 0
            start_proc_time = time.time()

            try:
                while cap.isOpened() and analyzed_idx < max_frames:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    raw_idx += 1
                    if raw_idx % sample_step != 0:
                        continue

                    analyzed_idx += 1
                    timestamp_s = raw_idx / fps

                    # Shared inference pipeline
                    res = engine.process_frame(frame, allow_direct_crop_fallback=False, draw_annotations=True)

                    # Update temporal decision engine
                    decision = temporal_engine.update(
                        eye_state_pred=res.overall_class,
                        confidence=res.overall_confidence,
                        current_time=timestamp_s,
                    )

                    curr_state = decision["state"]
                    closure_dur = decision["closure_duration"]
                    alert_now = decision["alert_triggered"]

                    if alert_now:
                        alert_events.append({
                            "Timestamp (s)": round(timestamp_s, 2),
                            "Closure Duration (s)": round(closure_dur, 2),
                            "PERCLOS (%)": round(decision["perclos"], 1),
                        })

                    frame_records.append({
                        "frame_idx": raw_idx,
                        "timestamp_s": round(timestamp_s, 2),
                        "overall_state": res.overall_state,
                        "pred_class": res.overall_class,
                        "confidence": res.overall_confidence,
                        "state": curr_state,
                        "closure_duration_s": round(closure_dur, 2),
                        "perclos_pct": decision["perclos"],
                        "alert": alert_now,
                    })

                    # Live Dashboard Update every 4 frames
                    if analyzed_idx % 4 == 0:
                        video_placeholder.image(cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                        progress = min(1.0, raw_idx / max(1, total_frames))
                        progress_bar.progress(progress)
                        status_text.caption(f"Processed frame {raw_idx}/{total_frames} ({progress:.0%}) | Time: {timestamp_s:.1f}s")

                        badge_kpi = "badge-awake" if curr_state == DrowsinessEngine.STATE_ALERT else ("badge-drowsy" if alert_now else "badge-warn")
                        kpi_state.markdown(f"**State:** <span class='status-badge {badge_kpi}'>{curr_state}</span>", unsafe_allow_html=True)
                        kpi_closure.markdown(f"**Closure Duration:** `{closure_dur:.2f}s` / `{closure_sec:.1f}s`")
                        kpi_blinks.markdown(f"**Total Blinks:** `{temporal_engine.total_blinks}`")
                        kpi_perclos.markdown(f"**Rolling PERCLOS:** `{decision['perclos']:.1f}%`")
                        if alert_now:
                            kpi_alert.error("🚨 DROWSINESS ALERT ACTIVE")
                        else:
                            kpi_alert.empty()
            finally:
                cap.release()
                if temp_file_to_clean and os.path.exists(temp_file_to_clean):
                    os.unlink(temp_file_to_clean)

            progress_bar.progress(1.0)
            status_text.caption("✅ Temporal Video Analysis Complete.")

            # Post-Analysis Summary
            if frame_records:
                df = pd.DataFrame(frame_records)
                st.markdown("---")
                st.markdown("### 📊 Comprehensive Telemetry & Safety Report")

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Analyzed Frames", f"{len(df):,}")
                r2.metric("Detected Blinks", temporal_engine.total_blinks)
                r3.metric("Drowsy Episodes Triggered", len(alert_events))
                r4.metric("Mean PERCLOS", f"{df['perclos_pct'].mean():.1f}%")

                # Charts
                tab_time, tab_dist = st.tabs(["📈 Eye Closure Timeline", "📊 State Distribution"])
                with tab_time:
                    st.line_chart(df.set_index("timestamp_s")["closure_duration_s"])
                    st.caption("Continuous ocular closure duration (seconds) plotted across video runtime.")

                with tab_dist:
                    st.bar_chart(df["state"].value_counts())
                    st.caption("Distribution of behavioral states identified across video frames.")

                # Prolonged Closure Events
                if alert_events:
                    st.markdown("#### 🚨 Logged Prolonged-Closure Episodes")
                    st.dataframe(pd.DataFrame(alert_events), use_container_width=True)
                else:
                    st.success("No sustained drowsiness episodes (≥ threshold) were detected during this video sequence.")

                # Telemetry Download
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Frame Telemetry Log (CSV)",
                    data=csv_bytes,
                    file_name=f"drowsiness_telemetry_{int(time.time())}.csv",
                    mime="text/csv",
                    type="primary",
                )


# ─── PAGE 4: Live Webcam ───
def render_webcam_page(engine: Optional[InferenceEngine]):
    st.markdown("<h2 style='margin-bottom: 4px;'>📹 Live Camera Monitoring</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 22px;'>Real-time browser camera snapshot diagnosis and native OpenCV live video stream monitor.</p>", unsafe_allow_html=True)

    if engine is None:
        st.error("Inference Engine unavailable. Please verify model checkpoint.")
        return

    tab_snap, tab_stream = st.tabs(["🌐 Browser Camera Snapshot", "🖥️ Native OpenCV Live Stream"])

    with tab_snap:
        st.markdown("""
        Capture a frame directly via your web browser to perform immediate dual-eye state classification.
        """)
        camera_img = st.camera_input("Take a camera snapshot for ocular analysis")

        if camera_img is not None:
            bytes_data = camera_img.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv_img is None:
                st.error("Could not decode camera image.")
                return

            # Run through the shared inference pipeline
            res = engine.process_frame(cv_img, allow_direct_crop_fallback=False, draw_annotations=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Annotated Camera Capture")
                st.image(cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                badge = "badge-awake" if res.overall_class == 0 else ("badge-closed" if res.overall_class == 1 else "badge-warn")
                st.markdown(f"### Status: <span class='status-badge {badge}'>{res.overall_state}</span>", unsafe_allow_html=True)
                st.write(f"• **Confidence:** `{res.overall_confidence:.1%}`")
                st.write(f"• **Eyes Localized:** `{res.eyes_detected_count} Regions`")
                st.write(f"• **Localization Method:** `{res.detection_method}`")
                st.write(f"• **Inference Latency:** `{res.latency_ms:.1f} ms`")

                if res.left_eye or res.right_eye:
                    st.markdown("---")
                    st.markdown("#### Bilateral Eye Analysis")
                    for eye in [res.left_eye, res.right_eye]:
                        if eye is not None:
                            st.write(f"**{eye.name}:** {eye.pred_label} (Conf: `{eye.confidence:.1%}`, Sigmoid: `{eye.raw_probability:.3f}`)")

    with tab_stream:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🖥️ Continuous OpenCV Webcam Stream with Acoustic Alerts</div>
            <div class="card-desc">
                For continuous real-time 30 FPS monitoring with acoustic alarm buzzer triggers, run the dedicated OpenCV application.
                It connects directly to your video hardware without browser sandbox limitations and triggers native Windows audio alerts on sustained closure.
            </div>
            <div class="formula-box" style="margin-top: 14px;">
                & ".\.venv\Scripts\python.exe" realtime_webcam.py --threshold 2.0
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin-top: 12px; line-height: 1.6;">
                <strong>Interactive Controls in native window:</strong><br>
                • Press <code>q</code>: Safely exit and release video capture hardware.<br>
                • Audio Alarm: Emits a 1200 Hz alert tone upon sustained closure ≥ 2.0 seconds.<br>
                • Overlay HUD: Displays continuous closure timer, live FPS, and bilateral ocular status.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─── PAGE 5: Project Overview ───
def render_project_overview():
    st.markdown("<h2 style='margin-bottom: 4px;'>📖 Project Overview & Problem Statement</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Comprehensive examination of driver fatigue detection through deep learning and temporal computer vision.</p>", unsafe_allow_html=True)

    st.markdown("""
    ### 1. Problem Statement
    Driver fatigue is implicated in over **20% of commercial and vehicular highway collisions** worldwide.
    Human microsleeps—episodes of involuntary eye closure lasting between 1.5 and 5 seconds—occur without conscious
    driver awareness. Detecting these events presents several engineering challenges:
    - **Distinguishing Blinks vs Drowsiness:** Normal human blinks last 100–400 ms. A single-frame image classifier cannot determine if closed eyes are a healthy blink or a lethal microsleep.
    - **Computational Constraints:** Vehicles and edge dashcams run constrained CPUs without dedicated NVIDIA GPUs.
    - **Environmental Heterogeneity:** Varying ambient sunlight, headlight glare, shadows, and spectacles with reflective lenses disrupt traditional heuristic formulas.

    ---

    ### 2. Project Objectives
    1. **Real-Time CPU Classification:** Deliver sub-10 ms inference on standard laptop CPUs using MobileNetV3 architectures.
    2. **Eye Localization:** Use FaceMesh when available, OpenCV Haar cascades otherwise, and manual crop selection for still images.
    3. **Temporal Filtering:** Implement continuous rolling timestamps and PERCLOS algorithms to eliminate single-frame false positives.
    4. **Unified Deployment:** Provide identical inference across static image uploads, batch video telemetry, and continuous webcam feeds.

    ---

    ### 3. Local Datasets & Their Architectural Roles
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">A. MRL Eye Dataset (Forked)</div>
            <div class="card-desc">
                <strong>Location:</strong> <code>archive/data/</code><br>
                <strong>Total Discovered:</strong> 84,898 grayscale infrared eye images.<br>
                <strong>Splits:</strong> Train: 50,937 | Val: 16,980 | Test: 16,981.<br>
                <strong>Class Balance:</strong> 42,952 Awake (50.6%) / 41,946 Sleepy (49.4%).<br>
                <strong>Subjects:</strong> 37 distinct human subjects (s0001 - s0037).<br>
                <strong>Role:</strong> Trains the MobileNetV3 model to become a high-precision infrared ocular feature specialist.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">B. UTA-RLDD Dataset</div>
            <div class="card-desc">
                <strong>Location:</strong> <code>UTA-RLDD Dataset/</code><br>
                <strong>Total Discovered:</strong> 11,787 face-level video frames.<br>
                <strong>Splits:</strong> Train: 9,054 | Val: 1,824 | Test: 909.<br>
                <strong>Classes:</strong> Active (Alert) vs Fatigue (Drowsy Video Session).<br>
                <strong>Role:</strong> Evaluates domain adaptation and cross-domain generalization from tight eye crops to whole-face video sequences.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── PAGE 6: System Architecture ───
def render_architecture_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>🏛️ System Architecture & Mathematical Specifications</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>The implemented path from camera or file input through eye-state prediction and, for sequences, temporal reporting.</p>", unsafe_allow_html=True)

    flow = st.columns(5)
    steps = [
        ("1 · Input", "Uploaded image, sampled video frame, camera snapshot, or local webcam frame"),
        ("2 · Locate", "MediaPipe FaceMesh if its API exists; otherwise OpenCV Haar face and eye cascades; manual ROI for still images"),
        ("3 · Prepare", "BGR to grayscale, resize to 128×128, replicate channels, scale to [-1, 1]"),
        ("4 · Classify", "MobileNetV3Small sigmoid output: P(closed); threshold 0.507"),
        ("5 · Summarize", "Still image: eye state. Video/webcam: temporal closure, blinks, alerts, PERCLOS"),
    ]
    for column, (title, description) in zip(flow, steps):
        column.markdown(f"<div class='custom-card' style='min-height:150px'><div class='card-title'>{title}</div><div class='card-desc'>{description}</div></div>", unsafe_allow_html=True)

    st.info("Training is a separate offline workflow: MRL eye crops are divided by subject into train, validation, and test partitions; the model is optimized on train, monitored on validation, then measured on test. The saved history below is shown exactly as recorded in `outputs/training_history.csv`.")

    st.markdown("---")
    st.subheader("🧠 Neural Network Architecture: MobileNetV3Small Backbone")

    st.markdown("""
    ```
    Input: (128, 128, 3) Normalized Image Tensor [-1.0, 1.0]
      │
      ▼
    MobileNetV3Small Backbone (ImageNet Pretrained)
      │  ├── 16 Inverted Residual Bottleneck Blocks with Depthwise Convolutions
      │  ├── Squeeze-and-Excitation (SE) Channel Attention Modules
      │  └── Output: (4, 4, 576) Feature Map
      ▼
    GlobalAveragePooling2D ──► (576-D Feature Vector)
      │
      ▼
    Dropout (rate = 0.30)
      │
      ▼
    Dense (64 units, activation = ReLU)
      │
      ▼
    Dropout (rate = 0.15)
      │
      ▼
    Dense (1 unit, activation = Sigmoid) ──► P(Sleepy) in [0.0, 1.0]
    ```
    """)

    st.markdown("---")
    st.subheader("📐 Activation Functions Mathematical Derivation")

    act_col1, act_col2 = st.columns(2)

    with act_col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">1. Hard-Swish (h-swish)</div>
            <div class="formula-box">
                h-swish(x) = x · [ ReLU6(x + 3) / 6 ]
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.55;">
                <strong>Where Used:</strong> The MobileNetV3Small backbone.<br>
                <strong>Mathematical Purpose:</strong> Eliminates gradient saturation on large positive values while remaining non-monotonic.<br>
                <strong>Computational Benefit:</strong> Replaces the expensive exponential term e<sup>-x</sup> of standard Swish with a piecewise linear formulation, saving critical CPU clock cycles without loss in ocular feature extraction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with act_col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">2. Rectified Linear Unit (ReLU)</div>
            <div class="formula-box">
                ReLU(x) = max(0, x)
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.55;">
                <strong>Where Used:</strong> The backbone and custom Dense(64) projection.<br>
                <strong>Mathematical Purpose:</strong> Enforces activation sparsity and provides a constant derivative f'(x) = 1 for positive activations, preventing vanishing gradients.<br>
                <strong>Computational Benefit:</strong> Efficiently compresses the 576-dimensional pooled vector into 64 bottleneck features, retaining only the most prominent ocular cues.
            </p>
        </div>
        """, unsafe_allow_html=True)

    act_col3, act_col4 = st.columns(2)

    with act_col3:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">3. Squeeze-and-Excitation (SE) Gating</div>
            <div class="formula-box">
                s = σ(W₂ · ReLU(W₁ · z))
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.55;">
                <strong>Where Used:</strong> MobileNetV3Small attention blocks.<br>
                <strong>Mathematical Purpose:</strong> Dynamically recalculates inter-channel dependencies via global average pooling and two pointwise convolutions.<br>
                <strong>Application:</strong> Reweights feature channels based on learned image features.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with act_col4:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">4. Sigmoid Output (σ)</div>
            <div class="formula-box">
                σ(z) = 1 / (1 + e<sup>-z</sup>)
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.55;">
                <strong>Where Used:</strong> Final single-unit output layer: <code>Dense(1, activation='sigmoid')</code>.<br>
                <strong>Mathematical Purpose:</strong> Maps the scalar logit to the model score interpreted as P(eye closed | x).<br>
                <strong>Decision Threshold:</strong> Class 1 (Sleepy) if P(Sleepy) ≥ 0.507 (validation-optimal threshold); otherwise Class 0 (Awake).
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─── PAGE 7: Model Training & Evaluation ───
def render_evaluation_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>📊 Training History & Evaluation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Curves are read from the saved epoch history. Confusion matrices and test metrics are read from the latest evaluation JSON.</p>", unsafe_allow_html=True)

    metrics_path = os.path.join(config.OUTPUTS_DIR, "evaluation.json")
    payload = {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        st.warning("No readable evaluation.json is available. Run `python evaluate.py` to generate current test results.")
    saved_metrics = payload.get("eye_state_classification", {})
    if saved_metrics.get("accuracy") is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MRL Test Accuracy", f"{saved_metrics['accuracy']:.2%}", help=f"{saved_metrics.get('test_samples', 0):,} eye crops")
        k2.metric("Sleepy Precision", f"{saved_metrics.get('precision', 0):.2%}")
        k3.metric("Sleepy Recall", f"{saved_metrics.get('recall', 0):.2%}")
        k4.metric("Sleepy F1", f"{saved_metrics.get('f1_score', 0):.2%}")
        st.caption(f"Evaluation generated: {payload.get('timestamp', 'timestamp unavailable')} · Decision threshold: {config.CONFIDENCE_THRESHOLD:.3f} · Current checkpoint: {os.path.basename(config.EYE_STATE_MODEL_PATH)}")
        st.info("The MRL evaluation subset is subject-disjoint by the current splitter, but the existing checkpoint’s original training provenance is not tied to that split. Treat this as a measured local test result, not a confirmed unseen-driver result.")

    history_path = os.path.join(config.OUTPUTS_DIR, "training_history.csv")
    st.markdown("---")
    st.subheader("Training and Validation Curves")
    if os.path.isfile(history_path):
        try:
            history = pd.read_csv(history_path)
            needed = {"accuracy", "val_accuracy", "loss", "val_loss"}
            if not needed.issubset(history.columns) or history.empty:
                st.warning("The saved training history is missing accuracy/loss or validation columns.")
            else:
                history = history[list(needed)].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
                history.insert(0, "Epoch", np.arange(1, len(history) + 1))
                acc_chart, loss_chart = st.columns(2)
                with acc_chart:
                    st.markdown("**Accuracy**")
                    accuracy_plot = history[["Epoch", "accuracy", "val_accuracy"]].rename(columns={"accuracy": "Training", "val_accuracy": "Validation"})
                    st.line_chart(accuracy_plot, x="Epoch", y=["Training", "Validation"])
                with loss_chart:
                    st.markdown("**Loss**")
                    loss_plot = history[["Epoch", "loss", "val_loss"]].rename(columns={"loss": "Training", "val_loss": "Validation"})
                    st.line_chart(loss_plot, x="Epoch", y=["Training", "Validation"])
                st.caption(f"Showing {len(history)} recorded epochs from `{os.path.relpath(history_path, config.PROJECT_ROOT)}`. The artifact does not include a run ID linking it to the current checkpoint.")
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            st.warning(f"Could not read saved training history: {exc}")
    else:
        st.info("Training curves are not available yet. Run `python train.py` to write outputs/training_history.csv.")

    st.markdown("---")
    st.subheader("Confusion Matrices")
    st.caption("Rows are actual labels and columns are predicted labels. MRL order: Awake/open, then Sleepy/closed. UTA-RLDD order: Active, then Fatigue. The UTA result is a cross-domain check, not directly comparable to eye-state classification.")
    matrix_cols = st.columns(2)
    matrix_specs = [
        ("MRL eye-state test", saved_metrics, ("Awake", "Sleepy")),
        ("UTA-RLDD whole-face transfer", payload.get("video_face_drowsiness_analysis", {}), ("Active", "Fatigue")),
    ]
    for column, (title, metrics, labels) in zip(matrix_cols, matrix_specs):
        with column:
            st.markdown(f"**{title}**")
            matrix = metrics.get("confusion_matrix")
            if matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
                cm_df = pd.DataFrame(matrix, index=[f"Actual {labels[0]}", f"Actual {labels[1]}"], columns=[f"Pred {labels[0]}", f"Pred {labels[1]}"])
                st.dataframe(cm_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
                if title.startswith("MRL"):
                    st.caption(f"{metrics.get('test_samples', 0):,} eye crops · accuracy {metrics.get('accuracy', 0):.2%}")
                else:
                    st.caption(f"{metrics.get('evaluated_samples', 0):,} face frames · accuracy {metrics.get('accuracy', 0):.2%} · F1 {metrics.get('f1_score', 0):.2%}")
            else:
                st.info("This confusion matrix is not present in the latest evaluation file. Run `python evaluate.py`.")

    per_class = saved_metrics.get("per_class", {})
    if per_class:
        st.markdown("---")
        st.subheader("MRL Per-Class Scores")
        st.dataframe(pd.DataFrame([{"Class": label, **values} for label, values in per_class.items()]), use_container_width=True)


# ─── PAGE 8: Dataset Explorer ───
def render_dataset_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>🔍 Dataset Explorer & Metadata Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Verified statistical inventory, split distributions, and sample galleries for local datasets.</p>", unsafe_allow_html=True)

    tab_mrl, tab_uta = st.tabs(["👁️ MRL Eye Dataset", "👤 UTA-RLDD Dataset"])

    with tab_mrl:
        mrl_stats = DatasetInspector.get_mrl_statistics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Images", f"{mrl_stats['total_images']:,}")
        c2.metric("Train Split", f"{mrl_stats['splits']['train']['total']:,}")
        c3.metric("Val Split", f"{mrl_stats['splits']['val']['total']:,}")
        c4.metric("Test Split", f"{mrl_stats['splits']['test']['total']:,}")

        st.markdown("#### Class Balance Distribution")
        split_df = pd.DataFrame([
            {"Split": split.title(), "Awake (Open)": values["awake"], "Sleepy (Closed)": values["sleepy"]}
            for split, values in mrl_stats["splits"].items()
        ]).set_index("Split")
        st.bar_chart(split_df)

        st.markdown("#### Sample Eye Crops Gallery")
        samples = DatasetInspector.get_sample_images("mrl", "test", count_per_class=4)
        if samples:
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown("**Awake Samples (Eyes Open)**")
                sub_cols = st.columns(len(samples.get("awake", [])))
                for idx, img_p in enumerate(samples.get("awake", [])):
                    sub_cols[idx].image(img_p, use_container_width=True)
            with g_col2:
                st.markdown("**Sleepy Samples (Eyes Closed)**")
                sub_cols = st.columns(len(samples.get("sleepy", [])))
                for idx, img_p in enumerate(samples.get("sleepy", [])):
                    sub_cols[idx].image(img_p, use_container_width=True)

    with tab_uta:
        uta_stats = DatasetInspector.get_uta_rldd_statistics()
        u1, u2, u3 = st.columns(3)
        u1.metric("Total Extracted Frames", f"{uta_stats['total_frames']:,}")
        u2.metric("Active Frames", f"{uta_stats['total_active']:,}")
        u3.metric("Fatigue Frames", f"{uta_stats['total_fatigue']:,}")

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Dataset Characteristics</div>
            <div class="card-desc">
                UTA-RLDD consists of portrait video frames of participants in various alertness levels (Active, Low Vigilance, Fatigue).
                Unlike MRL cropped eyes, UTA-RLDD contains full face frames. The current evaluation applies the eye classifier directly to those frames as a cross-domain stress check; it does not represent the complete eye-crop inference pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── PAGE 9: About / Limitations ───
def render_about_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>ℹ️ About, Limitations & Ethical Guidelines</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Real-world operational constraints and responsible AI safety guidelines.</p>", unsafe_allow_html=True)

    st.markdown("""
    ### 1. Operational & Environmental Limitations
    Deploying eye-state monitoring systems in vehicular environments faces several physical challenges:
    - **Eyewear & Reflections:** Thick spectacle frames, tinted lenses, and anti-glare coatings can occlude the iris or create specular reflections.
    - **Extreme Head Pose:** Significant yaw or pitch can prevent the frontal Haar face and eye cascades from finding a usable crop. Manual ROI selection is available for uploaded still images.
    - **Involuntary Physiological Twitches:** Blepharospasms, bright sunlight squinting, and laughing cause transient eye narrowing; the temporal engine filters these out by requiring sustained closure.
    - **Rapid Illumination Shifts:** Changes in lighting can reduce Haar detection and classifier reliability. No illumination normalization is currently applied by the runtime detector.

    ---

    ### 2. Ethical Use & Safety Disclaimer
    - **Not a Certified Medical Device:** This system does not diagnose sleep disorders, sleep apnea, or narcolepsy.
    - **Driver Responsibility:** Drivers retain continuous legal responsibility for vehicle control. This software is an academic assistance prototype and cannot replace attentive driving.
    - **Privacy:** All inference runs locally in the environment hosting the server. Video uploads use temporary files that are automatically deleted after processing.
    """)


# ─── Application Main Entry Point ───
def main():
    engine = get_cached_inference_engine()
    selected_page = render_sidebar(engine)

    if selected_page == "🏠 Home / Dashboard":
        render_home_page()
    elif selected_page == "🖼️ Image Analysis":
        render_image_page(engine)
    elif selected_page == "🎬 Video Analysis":
        render_video_page(engine)
    elif selected_page == "📹 Live Webcam":
        render_webcam_page(engine)
    elif selected_page == "📖 Project Overview":
        render_project_overview()
    elif selected_page == "🏛️ System Architecture":
        render_architecture_page()
    elif selected_page == "📊 Model Training & Evaluation":
        render_evaluation_page()
    elif selected_page == "🔍 Dataset Explorer":
        render_dataset_page()
    elif selected_page == "ℹ️ About / Limitations":
        render_about_page()


if __name__ == "__main__":
    main()
