"""
Human Drowsiness Detection — Deep Learning Web Application.

A modern, professional computer vision dashboard for driver drowsiness detection.
Powered by:
- Drowsiness Prediction Dataset (DPD) from Kaggle
- Pretrained MobileNetV3Small Deep Learning Architecture
- Face-level fatigue classification (Active vs Fatigue)
- Temporal Decision Engine for sustained drowsiness tracking
- Shared inference pipeline across Image Upload, Video Analysis, and Live Webcam
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
    .badge-active {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .badge-fatigue {
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
    if not os.path.exists(config.MODEL_PATH):
        return None
    try:
        model = load_trained_model(config.MODEL_PATH)
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
                <span class="status-badge badge-active">Active</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 6px;">
                <span style="color: #94a3b8;">Architecture:</span>
                <span style="color: #38bdf8; font-weight: 600;">MobileNetV3Small</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 6px;">
                <span style="color: #94a3b8;">Dataset:</span>
                <span style="color: #34d399; font-weight: 600;">DPD 9,120 Images</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem;">
                <span style="color: #94a3b8;">Execution Target:</span>
                <span style="color: #a78bfa; font-weight: 600;">Optimized CPU</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("⚠️ Model not found at models/drowsiness_model.keras")

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
        Model: MobileNetV3 (ImageNet Pretrained)<br>
        Dataset: DPD (Active vs Fatigue)
    </div>
    """, unsafe_allow_html=True)

    return selected_page


# ─── PAGE 1: Home / Dashboard ───
def render_home_page():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Human Drowsiness Detection</div>
        <div class="hero-subtitle">
            A computer vision demonstration that predicts the DPD directory labels <strong>Active</strong> or 
            <strong>Fatigue</strong> from face crops and summarizes prediction patterns over time.
            Powered by the <strong>Kaggle Drowsiness Prediction Dataset (DPD)</strong> and an ImageNet-pretrained 
            <strong>MobileNetV3Small</strong> deep learning classifier.
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
                Analyze driver portrait photos or cropped face images. Automatically locates face regions with multi-tier
                Haar cascades, classifies fatigue, and provides quantitative confidence telemetry.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🎬 Video Sequence Analysis</div>
            <div class="card-desc">
                Process driving video recordings (.mp4, .avi, .mov). Calculates continuous temporal fatigue duration,
                rolling fatigue percentage, timestamped drowsiness event logs, and downloadable telemetry CSVs.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">📹 Live Camera Monitoring</div>
            <div class="card-desc">
                Capture a browser snapshot for face-level classification or launch the dedicated local continuous OpenCV webcam stream
                featuring heads-up telemetry HUD and native acoustic warning alerts.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Real System Status Metrics
    st.markdown("<h3 style='margin-top: 24px; margin-bottom: 14px;'>⚡ Verified Platform Status</h3>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    has_model = os.path.isfile(config.MODEL_PATH)
    dpd_stats = DatasetInspector.get_dpd_statistics()

    m1.metric("Neural Checkpoint", "Ready" if has_model else "Missing")
    m2.metric("Target Architecture", "MobileNetV3Small")
    m3.metric("DPD Dataset Images", f"{dpd_stats['total_images']:,} Images")
    m4.metric("Directory Label Counts", dpd_stats["balance"])

    # Workflow Visualization
    st.markdown("<h3 style='margin-top: 28px; margin-bottom: 14px;'>🔬 Overall System Workflow</h3>", unsafe_allow_html=True)
    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #38bdf8; font-weight: 700; margin-bottom: 8px;">1. Image / Video Frame</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                Uploaded stills, video frames, and webcam feeds enter the unified inference engine as RGB tensors.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #818cf8; font-weight: 700; margin-bottom: 8px;">2. Face Localization</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                Ensemble Haar cascade detectors locate the driver's face, extract a padded bounding crop, or accept direct cropped faces.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #34d399; font-weight: 700; margin-bottom: 8px;">3. CNN Fatigue Classification</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                The face crop is resized to 128×128×3 RGB, scaled to [-1, 1], and classified by MobileNetV3 as Active (0) or Fatigue (1).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step4:
        st.markdown("""
        <div style="background: #0b1322; border: 1px solid #1a2a44; border-radius: 12px; padding: 18px; height: 100%;">
            <div style="color: #f59e0b; font-weight: 700; margin-bottom: 8px;">4. Temporal Decision Engine</div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin: 0;">
                Sequential predictions are smoothed over time. Sustained fatigue exceeding 2.0s triggers alarms and logs safety events.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ Non-Diagnostic Engineering Prototype Notice:</strong>
        This application is an experimental deep learning computer vision system. Facial fatigue indicators serve as situational
        safety assists, not certified medical diagnoses of clinical sleep apnea or autonomous vehicular safety compliance.
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE 2: Single Image Analysis ───
def render_image_page(engine: Optional[InferenceEngine]):
    st.markdown("<h2 style='margin-bottom: 4px;'>🖼️ Single Image Fatigue Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 22px;'>Analyze driver portraits, snapshots, or pre-cropped face images with multi-tier detection and neural fatigue scoring.</p>", unsafe_allow_html=True)

    if engine is None:
        st.error("Inference Engine unavailable. Please verify model checkpoint.")
        return

    # Quick Samples Bar
    st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 8px;'>🧪 Load Bundled Test Sample:</p>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    sample_path = None

    if b1.button("👤 Active Driver Face (Sample)", use_container_width=True):
        sample_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_face_active.jpg")
    if b2.button("🥱 Fatigue Driver Face (Sample)", use_container_width=True):
        sample_path = os.path.join(config.PROJECT_ROOT, "sample_data", "sample_face_fatigue.jpg")

    # Upload & Controls Row
    col_up, col_ctrl = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Image File (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            help="Upload a driver photograph or cropped facial image."
        )

    with col_ctrl:
        st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #cbd5e1; margin-bottom: 4px;'>⚙️ Processing Strategy:</p>", unsafe_allow_html=True)
        analysis_mode = st.radio(
            "Detection Mode",
            ["🤖 Auto-Detect Face (Full Frame)", "👤 Direct Face Crop (Skip Detection)"],
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

        # Run unified inference
        is_direct = (analysis_mode == "👤 Direct Face Crop (Skip Detection)")
        res: ImageAnalysisResult = engine.process_frame(
            image_to_process,
            allow_direct_face_fallback=is_direct,
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
                <span class="telemetry-chip">🎨 RGB 3-Channel</span>
                <span class="telemetry-chip">Mode: {analysis_mode}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_out:
            st.markdown("#### 🧠 Neural Detection Overlay")
            st.image(cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Determine badge styling
            if res.overall_class == 0:
                badge_class = "badge-active"
            elif res.overall_class == 1:
                badge_class = "badge-fatigue"
            else:
                badge_class = "badge-warn"

            st.markdown(f"""
            <div style="margin-top: 8px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600;">Status:</span>
                    <span class="status-badge {badge_class}" style="margin-left: 8px; font-size: 0.92rem;">{res.overall_state}</span>
                </div>
                <span class="telemetry-chip">Method: {res.detection_method}</span>
            </div>
            """, unsafe_allow_html=True)

        # Telemetry KPI Row
        st.markdown("<h4 style='margin-top: 24px;'>📊 Quantitative Fatigue Telemetry</h4>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Prediction", res.overall_state)
        k2.metric("Neural Confidence", f"{res.overall_confidence:.1%}")
        k3.metric("Face Detected", "Yes" if res.face_detected else "No")
        k4.metric("Inference Latency", f"{res.latency_ms:.1f} ms")

        # Face Result Card
        if res.face_result is not None:
            st.markdown("#### 👤 Facial Classification Details")
            c_f1, c_f2 = st.columns([1, 2])
            with c_f1:
                if res.face_result.crop_bgr.size > 0:
                    st.image(cv2.cvtColor(res.face_result.crop_bgr, cv2.COLOR_BGR2RGB), caption="Cropped Face Region", use_container_width=True)
            with c_f2:
                st.write(f"• **Predicted Class:** `{res.face_result.pred_label}`")
                st.write(f"• **Confidence:** `{res.face_result.confidence:.1%}`")
                st.write(f"• **Raw Probability Score:** `{res.face_result.raw_probability:.4f}` (0.0 = Active, 1.0 = Fatigue)")
                st.write(f"• **Face Bounding Box:** `{res.face_result.bbox}`")

        if not res.face_detected and not res.is_direct_face_image:
            st.warning("⚠️ " + res.message)


# ─── PAGE 3: Video Sequence Analysis ───
def render_video_page(engine: Optional[InferenceEngine]):
    st.markdown("<h2 style='margin-bottom: 4px;'>🎬 Video Sequence Drowsiness Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 22px;'>Continuous temporal evaluation calculating frame-by-frame fatigue duration, rolling fatigue ratio, and safety alerts.</p>", unsafe_allow_html=True)

    if engine is None:
        st.error("Inference Engine unavailable. Please verify model checkpoint.")
        return

    # Configuration Parameters
    with st.expander("⚙️ Temporal Analysis Parameters & Sensitivity", expanded=False):
        c1, c2, c3 = st.columns(3)
        sample_step = c1.slider("Sampling Step (Process every Nth frame)", 1, 6, int(config.VIDEO_FRAME_SAMPLE_RATE), help="Optimizes CPU processing throughput.")
        fatigue_sec = c2.slider("Drowsiness Alert Threshold (seconds)", 1.0, 4.0, float(config.FATIGUE_THRESHOLD_SEC), 0.2, help="Continuous fatigue duration triggering an alert.")
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
            temporal_engine = DrowsinessEngine(fatigue_threshold_sec=fatigue_sec)
            cap = cv2.VideoCapture(decode_path)

            col_disp, col_kpi = st.columns([3, 2])
            with col_disp:
                video_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

            with col_kpi:
                st.markdown("#### ⚡ Live Video Telemetry")
                kpi_state = st.empty()
                kpi_fatigue = st.empty()
                kpi_ratio = st.empty()
                kpi_events = st.empty()
                kpi_alert = st.empty()

            frame_records = []
            alert_events = []
            raw_idx = 0
            analyzed_idx = 0

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
                    res = engine.process_frame(frame, allow_direct_face_fallback=False, draw_annotations=True)

                    # Update temporal decision engine
                    decision = temporal_engine.update(
                        pred_class=res.overall_class,
                        confidence=res.overall_confidence,
                        current_time=timestamp_s,
                    )

                    curr_state = decision["state"]
                    fatigue_dur = decision["fatigue_duration"]
                    alert_now = decision["alert_triggered"]

                    if alert_now:
                        alert_events.append({
                            "Timestamp (s)": round(timestamp_s, 2),
                            "Fatigue Duration (s)": round(fatigue_dur, 2),
                            "Fatigue Ratio (%)": round(decision["fatigue_ratio"], 1),
                        })

                    frame_records.append({
                        "frame_idx": raw_idx,
                        "timestamp_s": round(timestamp_s, 2),
                        "overall_state": res.overall_state,
                        "pred_class": res.overall_class,
                        "confidence": res.overall_confidence,
                        "state": curr_state,
                        "fatigue_duration_s": round(fatigue_dur, 2),
                        "fatigue_ratio_pct": decision["fatigue_ratio"],
                        "alert": alert_now,
                    })

                    # Live Dashboard Update every 4 frames
                    if analyzed_idx % 4 == 0:
                        video_placeholder.image(cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                        progress = min(1.0, raw_idx / max(1, total_frames))
                        progress_bar.progress(progress)
                        status_text.caption(f"Processed frame {raw_idx}/{total_frames} ({progress:.0%}) | Time: {timestamp_s:.1f}s")

                        badge_kpi = "badge-active" if curr_state == DrowsinessEngine.STATE_ALERT else ("badge-drowsy" if alert_now else "badge-warn")
                        kpi_state.markdown(f"**State:** <span class='status-badge {badge_kpi}'>{curr_state}</span>", unsafe_allow_html=True)
                        kpi_fatigue.markdown(f"**Continuous Fatigue:** `{fatigue_dur:.2f}s` / `{fatigue_sec:.1f}s`")
                        kpi_ratio.markdown(f"**Rolling Fatigue Ratio:** `{decision['fatigue_ratio']:.1f}%`")
                        kpi_events.markdown(f"**Drowsy Events:** `{temporal_engine.total_drowsy_events}`")
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
                r2.metric("Mean Fatigue Ratio", f"{df['fatigue_ratio_pct'].mean():.1f}%")
                r3.metric("Drowsy Episodes Triggered", len(alert_events))
                r4.metric("Total Drowsy Events", temporal_engine.total_drowsy_events)

                # Charts
                tab_time, tab_dist = st.tabs(["📈 Fatigue Timeline", "📊 State Distribution"])
                with tab_time:
                    st.line_chart(df.set_index("timestamp_s")["fatigue_duration_s"])
                    st.caption("Continuous fatigue duration (seconds) plotted across video runtime.")

                with tab_dist:
                    st.bar_chart(df["state"].value_counts())
                    st.caption("Distribution of behavioral states identified across video frames.")

                # Prolonged Closure Events
                if alert_events:
                    st.markdown("#### 🚨 Logged Prolonged-Drowsiness Episodes")
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
        st.markdown("Capture a frame directly via your web browser to perform immediate face-level fatigue classification.")
        camera_img = st.camera_input("Take a camera snapshot for fatigue analysis")

        if camera_img is not None:
            bytes_data = camera_img.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv_img is None:
                st.error("Could not decode camera image.")
                return

            res = engine.process_frame(cv_img, allow_direct_face_fallback=False, draw_annotations=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Annotated Camera Capture")
                st.image(cv2.cvtColor(res.annotated_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                badge = "badge-active" if res.overall_class == 0 else ("badge-fatigue" if res.overall_class == 1 else "badge-warn")
                st.markdown(f"### Status: <span class='status-badge {badge}'>{res.overall_state}</span>", unsafe_allow_html=True)
                st.write(f"• **Confidence:** `{res.overall_confidence:.1%}`")
                st.write(f"• **Face Detected:** `{'Yes' if res.face_detected else 'No'}`")
                st.write(f"• **Detection Method:** `{res.detection_method}`")
                st.write(f"• **Inference Latency:** `{res.latency_ms:.1f} ms`")

    with tab_stream:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🖥️ Continuous OpenCV Webcam Stream with Acoustic Alerts</div>
            <div class="card-desc">
                For continuous real-time 30 FPS monitoring with acoustic alarm buzzer triggers, run the dedicated OpenCV application.
                It connects directly to your video hardware without browser sandbox limitations and triggers native Windows audio alerts on sustained fatigue.
            </div>
            <div class="formula-box" style="margin-top: 14px;">
                & ".\.venv\Scripts\python.exe" realtime_webcam.py --threshold 2.0
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin-top: 12px; line-height: 1.6;">
                <strong>Interactive Controls in native window:</strong><br>
                • Press <code>q</code>: Safely exit and release video capture hardware.<br>
                • Audio Alarm: Emits a 1200 Hz alert tone upon sustained fatigue ≥ 2.0 seconds.<br>
                • Overlay HUD: Displays continuous fatigue timer, live FPS, and face status.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─── PAGE 5: Project Overview ───
def render_project_overview():
    st.markdown("<h2 style='margin-bottom: 4px;'>📖 Project Overview & Problem Statement</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Comprehensive examination of driver fatigue detection through deep learning and temporal computer vision.</p>", unsafe_allow_html=True)

    st.markdown("""
    ### 1. Problem Statement
    Fatigue can reduce driver attention and response. This project explores face-level fatigue classification and temporal alerts as an engineering demonstration. It does not infer a medical state or establish road-safety performance. The main challenges include:
    - **Distinguishing Normal Lapses vs True Drowsiness:** Brief expressions or glances must not trigger jarring false alarms.
    - **Computational Constraints:** Vehicles and edge dashcams run constrained CPUs without dedicated NVIDIA GPUs.
    - **Environmental Heterogeneity:** Varying ambient sunlight, headlight glare, shadows, and spectacles with reflective lenses disrupt traditional heuristic formulas.

    ---

    ### 2. Dataset Migration to DPD (Kaggle)
    This project has completed a full migration to the **Drowsiness Prediction Dataset (DPD)** from Kaggle:
    - **Dataset Source:** [Kaggle Drowsiness Prediction Dataset](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset)
    - **Image Modality:** Full-size RGB portraits at varied resolutions; training and inference use the same padded face crop before resizing to 128×128.
    - **Directory Labels:**
        - `Active Subjects/`: 4,560 images (directory label: Active)
        - `Fatigue Subjects/`: 4,560 images (directory label: Fatigue)
    - **Label protocol:** The folder labels are used as supplied; the dataset page did not expose a detailed annotation protocol during this audit.
    - **Crop Coverage:** 9,057 images had a face crop; 63 detector misses are excluded and recorded.
    - **Old Datasets Removed:** The legacy MRL Eye and UTA-RLDD datasets have been fully deprecated and removed from the active training and evaluation pipelines.
    """)


# ─── PAGE 6: System Architecture ───
def render_architecture_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>🏛️ System Architecture & Mathematical Specifications</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>The implemented path from camera or file input through face-level fatigue prediction and temporal reporting.</p>", unsafe_allow_html=True)

    flow = st.columns(5)
    steps = [
        ("1 · Input", "Uploaded photo, sampled video frame, camera snapshot, or live webcam frame"),
        ("2 · Locate", "OpenCV Haar cascade ensemble (Frontal + Profile) with CLAHE equalization; or direct face crop"),
        ("3 · Prepare", "Resize to 128×128×3 RGB, normalize pixel range to [-1.0, 1.0]"),
        ("4 · Classify", "MobileNetV3Small sigmoid output: P(Fatigue); threshold 0.50"),
        ("5 · Summarize", "Still image: Active vs Fatigue state. Video/webcam: temporal fatigue, alerts, rolling ratio"),
    ]
    for column, (title, description) in zip(flow, steps):
        column.markdown(f"<div class='custom-card' style='min-height:150px'><div class='card-title'>{title}</div><div class='card-desc'>{description}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧠 Neural Network Architecture: MobileNetV3Small Backbone")

    st.markdown("""
    ```
    Input: (128, 128, 3) Normalized RGB Image Tensor [-1.0, 1.0]
      │
      ▼
    MobileNetV3Small Backbone (ImageNet Pretrained)
      │  ├── Inverted Residual Bottleneck Blocks with Depthwise Convolutions
      │  ├── Squeeze-and-Excitation (SE) Channel Attention Modules
      │  └── Output Feature Map
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
    Dense (1 unit, activation = Sigmoid) ──► P(Fatigue) in [0.0, 1.0]
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
                <strong>Where Used:</strong> MobileNetV3Small backbone.<br>
                <strong>Mathematical Purpose:</strong> Eliminates gradient saturation on large positive values while remaining non-monotonic.<br>
                <strong>Computational Benefit:</strong> Replaces exponential terms with piecewise linear operations, maximizing CPU speed.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with act_col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">2. Sigmoid Output (σ)</div>
            <div class="formula-box">
                σ(z) = 1 / (1 + e<sup>-z</sup>)
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.55;">
                <strong>Where Used:</strong> Final output layer: <code>Dense(1, activation='sigmoid')</code>.<br>
                <strong>Mathematical Purpose:</strong> Maps scalar logits to a fatigue score between 0 and 1.<br>
                <strong>Decision Rule:</strong> Class 1 (Fatigue) if P(Fatigue) ≥ 0.50; otherwise Class 0 (Active).
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─── PAGE 7: Model Training & Evaluation ───
def render_evaluation_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>📊 Training History & Model Evaluation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Measured metrics, confusion matrices, and training curves for the DPD-trained MobileNetV3 model.</p>", unsafe_allow_html=True)

    metrics_path = os.path.join(config.OUTPUTS_DIR, "evaluation.json")
    payload = {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        st.warning("No readable evaluation.json is available. Run `python evaluate.py` to generate current test results.")

    saved_metrics = payload.get("evaluation_metrics", {})
    if saved_metrics.get("accuracy") is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("DPD Test Accuracy", f"{saved_metrics['accuracy']:.2%}", help=f"{saved_metrics.get('test_samples', 0):,} face test samples")
        k2.metric("Fatigue Precision", f"{saved_metrics.get('precision', 0):.2%}")
        k3.metric("Fatigue Recall", f"{saved_metrics.get('recall', 0):.2%}")
        k4.metric("Fatigue F1 Score", f"{saved_metrics.get('f1_score', 0):.2%}")

        m_k1, m_k2 = st.columns(2)
        m_k1.metric("ROC AUC Score", f"{saved_metrics.get('roc_auc', 0):.4f}")
        m_k2.metric("Batched Test Throughput", f"{saved_metrics.get('cpu_fps', 0):.1f} FPS")
        st.caption("Measured on the batched test pipeline; this is not end-to-end face-detection or live-webcam FPS.")
        st.caption(f"Evaluation generated: {payload.get('timestamp', 'timestamp unavailable')} · Checkpoint: {os.path.basename(config.MODEL_PATH)}")

    history_path = os.path.join(config.OUTPUTS_DIR, "training_history.csv")
    st.markdown("---")
    st.subheader("Training and Validation Curves")
    if os.path.isfile(history_path):
        try:
            history = pd.read_csv(history_path)
            needed = {"accuracy", "val_accuracy", "loss", "val_loss"}
            if needed.issubset(history.columns) and not history.empty:
                history = history[list(needed)].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
                history.insert(0, "Epoch", np.arange(1, len(history) + 1))
                acc_chart, loss_chart = st.columns(2)
                with acc_chart:
                    st.markdown("**Accuracy over Epochs**")
                    accuracy_plot = history[["Epoch", "accuracy", "val_accuracy"]].rename(columns={"accuracy": "Training", "val_accuracy": "Validation"})
                    st.line_chart(accuracy_plot, x="Epoch", y=["Training", "Validation"])
                with loss_chart:
                    st.markdown("**Loss over Epochs**")
                    loss_plot = history[["Epoch", "loss", "val_loss"]].rename(columns={"loss": "Training", "val_loss": "Validation"})
                    st.line_chart(loss_plot, x="Epoch", y=["Training", "Validation"])
        except Exception as exc:
            st.warning(f"Could not read saved training history: {exc}")
    else:
        st.info("Training curves will appear once training completes.")

    # Confusion matrix and ROC
    st.markdown("---")
    st.subheader("Confusion Matrix & Visualizations")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        if saved_metrics.get("confusion_matrix"):
            cm_df = pd.DataFrame(
                saved_metrics["confusion_matrix"],
                index=["Actual Active", "Actual Fatigue"],
                columns=["Pred Active", "Pred Fatigue"],
            )
            st.dataframe(cm_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
            st.caption("Held-out DPD test images · rows are actual labels; columns are predictions.")
        else:
            st.info("The confusion matrix will appear after `python evaluate.py` completes.")

    with col_roc:
        roc_points = saved_metrics.get("roc_curve_points", [])
        if roc_points:
            roc_df = pd.DataFrame(roc_points).rename(columns={
                "false_positive_rate": "False Positive Rate",
                "true_positive_rate": "Model TPR",
            })
            roc_df["Random Chance"] = roc_df["False Positive Rate"]
            st.line_chart(roc_df, x="False Positive Rate", y=["Model TPR", "Random Chance"])
            st.caption("ROC points use the held-out DPD test predictions.")
        else:
            st.info("The ROC curve will appear after `python evaluate.py` completes.")

    per_class = saved_metrics.get("per_class", {})
    if per_class:
        st.markdown("---")
        st.subheader("Per-Class Detailed Performance")
        st.dataframe(pd.DataFrame([{"Class": label, **values} for label, values in per_class.items()]), use_container_width=True)


# ─── PAGE 8: Dataset Explorer ───
def render_dataset_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>🔍 Dataset Explorer & Metadata Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Statistical breakdown and sample gallery of the Kaggle Drowsiness Prediction Dataset (DPD).</p>", unsafe_allow_html=True)

    dpd_stats = DatasetInspector.get_dpd_statistics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Images", f"{dpd_stats['total_images']:,}")
    c2.metric("Active Label Images", f"{dpd_stats['total_active']:,}")
    c3.metric("Fatigue Label Images", f"{dpd_stats['total_fatigue']:,}")
    c4.metric("Native Resolution", dpd_stats.get("native_resolution", "128×128"))
    st.caption(
        f"Directory labels only; face crop cache covers {dpd_stats['total_images'] - dpd_stats.get('face_crop_excluded', 0):,} of "
        f"{dpd_stats['total_images']:,} images. {dpd_stats.get('face_crop_excluded', 0)} images were excluded because no face crop was found."
    )

    st.markdown("#### Split Distribution")
    split_df = pd.DataFrame([
        {"Split": split.title(), "Active": values["active"], "Fatigue": values["fatigue"]}
        for split, values in dpd_stats["splits"].items()
    ]).set_index("Split")
    st.bar_chart(split_df)

    st.markdown("#### Sample Face Gallery")
    samples = DatasetInspector.get_sample_images(count_per_class=4)
    if samples:
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("**Active Subjects (directory label: Active)**")
            sub_cols = st.columns(len(samples.get("Active", [])))
            for idx, img_p in enumerate(samples.get("Active", [])):
                sub_cols[idx].image(img_p, use_container_width=True)
        with g_col2:
            st.markdown("**Fatigue Subjects (directory label: Fatigue)**")
            sub_cols = st.columns(len(samples.get("Fatigue", [])))
            for idx, img_p in enumerate(samples.get("Fatigue", [])):
                sub_cols[idx].image(img_p, use_container_width=True)


# ─── PAGE 9: About / Limitations ───
def render_about_page():
    st.markdown("<h2 style='margin-bottom: 4px;'>ℹ️ About, Limitations & Ethical Guidelines</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 24px;'>Real-world operational constraints and responsible AI safety guidelines.</p>", unsafe_allow_html=True)

    st.markdown("""
    ### 1. Operational & Environmental Limitations
    Deploying facial drowsiness detection systems in vehicular cabins faces several physical challenges:
    - **Eyewear & Reflections:** Thick spectacle frames and tinted lenses can occlude subtle ocular signs.
    - **Extreme Head Pose:** Significant head rotation can prevent frontal Haar face cascades from detecting a face.
    - **Rapid Illumination Shifts:** Night driving, oncoming headlights, and tunnels cause severe contrast swings.
    - **Transient Expressions:** Yawning, laughing, or talking can momentarily resemble fatigue; the temporal decision engine filters these via continuous duration thresholds.

    ---

    ### 2. Ethical Use & Safety Disclaimer
    - **Not a Medical Device:** This system does not diagnose clinical sleep disorders or replace vehicle operator attentiveness.
    - **Driver Responsibility:** Operators maintain continuous legal responsibility for vehicle control at all times.
    - **Local Processing & Privacy:** All neural inference runs entirely locally on CPU; no image data leaves the device.
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
