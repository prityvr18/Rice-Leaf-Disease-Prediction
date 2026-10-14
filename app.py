from pathlib import Path
import hashlib
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "mobilenetv2___transfer_and_augmentation.keras"
CLASS_DATA_PATH = OUTPUT_DIR / "image_analysis.csv"
METRICS_PATH = OUTPUT_DIR / "model_comparison.csv"
IMAGE_ROOT = PROJECT_DIR / "data" / "extracted"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}

st.set_page_config(
    page_title="RiceLeaf | Leaf health scan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #18342a;
        --muted: #6c8075;
        --green: #286449;
        --soft-green: #eaf2e9;
        --surface: #fffefa;
        --line: #e8ece5;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 0%, rgba(212, 228, 192, .42), transparent 29rem),
            #f6f7f2;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.035em;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
    [data-testid="stSidebar"] { background: #f0f3eb; border-right: 1px solid var(--line); }
    .eyebrow {
        color: var(--green); font-size: .76rem; font-weight: 700; letter-spacing: .16em;
        text-transform: uppercase; margin-bottom: .65rem;
    }
    .hero-title {
        color: var(--ink); font: 800 clamp(2.5rem, 5vw, 4.5rem)/1.02 'Manrope', sans-serif;
        letter-spacing: -.065em; margin: .2rem 0 .9rem;
    }
    .hero-copy { color: var(--muted); font-size: 1.08rem; max-width: 620px; line-height: 1.7; }
    .brand-row { display: flex; align-items: center; gap: 1.1rem; margin: .4rem 0 .8rem; }
    .brand-mark {
        display: grid; place-items: center; flex: 0 0 96px; width: 96px; height: 96px;
        border: 1px solid #d8e6d2; border-radius: 28px;
        background: linear-gradient(145deg, #edf5e8, #dcebd5);
        box-shadow: 0 12px 30px rgba(40, 100, 73, .12);
        font-size: 3.9rem; line-height: 1;
    }
    .feature-photo {
        overflow: hidden; border: 1px solid var(--line); border-radius: 24px;
        background: #fffefa; box-shadow: 0 14px 34px rgba(35, 61, 44, .08);
    }
    .feature-caption { color: var(--muted); font-size: .78rem; margin-top: .45rem; text-align: right; }
    .section-kicker { color: var(--green); font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .panel {
        background: var(--surface); border: 1px solid var(--line); border-radius: 22px;
        padding: 1.25rem 1.4rem; box-shadow: 0 12px 34px rgba(35, 61, 44, .045);
    }
    .result-card {
        background: linear-gradient(140deg, #eaf3e7, #f8faf3);
        border: 1px solid #dce9d8; border-radius: 22px; padding: 1.6rem;
    }
    .result-label { color: var(--muted); font-size: .83rem; font-weight: 600; text-transform: uppercase; letter-spacing: .09em; }
    .result-name { color: var(--ink); font: 800 clamp(1.7rem, 3vw, 2.45rem)/1.15 'Manrope', sans-serif; margin: .45rem 0 .65rem; }
    .confidence-pill {
        display: inline-block; padding: .42rem .78rem; border-radius: 999px;
        background: #dcebd8; color: #255e42; font-size: .85rem; font-weight: 700;
    }
    .result-card h3 { margin: .2rem 0 .45rem; }
    .small-note { color: var(--muted); font-size: .85rem; line-height: 1.55; }
    [data-testid="stMetric"] {
        background: #fffefa; border: 1px solid var(--line); border-radius: 16px;
        padding: .95rem 1rem; box-shadow: 0 8px 24px rgba(35, 61, 44, .04);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); overflow-wrap: anywhere; font-size: 1.25rem; }
    div[data-testid="stFileUploader"] section { background: #fffefa; border-radius: 16px; border-color: #dce5d9; }
    .stButton > button[kind="primary"] {
        background: #286449; border: 0; border-radius: 12px; min-height: 3rem;
        font-weight: 700; box-shadow: 0 6px 16px rgba(40, 100, 73, .18);
    }
    .stButton > button[kind="primary"]:hover { background: #1f513a; border: 0; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_class_metadata() -> pd.DataFrame:
    if not CLASS_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Class metadata is missing: {CLASS_DATA_PATH}. Run all cells in the project notebook first."
        )

    metadata = pd.read_csv(CLASS_DATA_PATH)
    required_columns = {"label", "class"}
    if not required_columns.issubset(metadata.columns):
        raise ValueError("image_analysis.csv must include 'label' and 'class' columns.")

    classes = metadata[["label", "class"]].drop_duplicates().sort_values("label")
    if classes["label"].tolist() != list(range(len(classes))):
        raise ValueError("Class metadata must have contiguous labels starting from zero.")
    return classes.reset_index(drop=True)


@st.cache_data
def get_feature_image() -> Optional[np.ndarray]:
    image_paths = sorted(IMAGE_ROOT.glob("**/*.jpg"))
    if not image_paths:
        image_paths = sorted(IMAGE_ROOT.glob("**/*.JPG"))
    if not image_paths:
        return None

    bgr_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if bgr_image is None:
        return None
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner="Loading the trained leaf classifier...")
def load_classifier(model_path: str, modified_time: float) -> tf.keras.Model:
    del modified_time
    return tf.keras.models.load_model(model_path, compile=False)


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("This file could not be decoded as a JPEG or PNG image.")
    height, width = bgr_image.shape[:2]
    if height * width > 20_000_000:
        raise ValueError("Image dimensions are too large. Please upload an image under 20 megapixels.")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def predict_image(model: tf.keras.Model, rgb_image: np.ndarray, labels: list[str]) -> np.ndarray:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        raise ValueError("The saved model unexpectedly has multiple inputs.")
    height, width = input_shape[1:3]
    if height is None or width is None:
        raise ValueError(f"The model has a dynamic input size and cannot be used here: {input_shape}")

    resized = cv2.resize(rgb_image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    prediction = model.predict(np.expand_dims(resized.astype(np.float32), axis=0), verbose=0)
    scores = np.asarray(prediction)[0]
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError(
            f"Model returned {len(scores)} outputs but class metadata has {len(labels)} labels."
        )
    if not np.isfinite(scores).all():
        raise ValueError("The model returned invalid prediction scores.")
    return scores


with st.sidebar:
    st.markdown("## 🌾 RiceLeaf")
    st.caption("A small-data computer vision project")
    st.markdown("---")
    st.markdown("**Model**")
    st.write("MobileNetV2 · transfer learning")
    st.write("ImageNet features · 3 rice-leaf classes")
    st.markdown("---")
    st.markdown("**How to use**")
    st.write("Upload a clear JPEG or PNG leaf photo, then select **Analyze image**.")
    st.markdown("---")
    st.caption("Educational project · not a diagnosis or treatment recommendation.")

hero_copy, hero_photo_column = st.columns([1.35, 0.85], gap="large")
with hero_copy:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark" role="img" aria-label="Rice plant logo">🌾</div>
            <div>
                <div class="eyebrow">FIELD NOTES · RICE PLANT HEALTH</div>
                <div class="hero-title">RiceLeaf<br>health studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy">A closer look at your rice leaves. Upload a photo to see how the '
        'project model classifies it.</p>',
        unsafe_allow_html=True,
    )
with hero_photo_column:
    featured_image = get_feature_image()
    if featured_image is not None:
        st.image(featured_image, use_column_width=True, caption="A leaf from the project dataset")
    else:
        st.markdown(
            '<div class="panel"><div class="section-kicker">RICE LEAF IMAGE CLASSIFIER</div>'
            '<h3>Upload a leaf photo to begin</h3></div>',
            unsafe_allow_html=True,
        )
st.write("")

try:
    class_metadata = load_class_metadata()
    class_names = class_metadata["class"].astype(str).tolist()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Cannot load the trained model's class labels: {error}")
    st.info("Run all cells in RiceLeaf_Disease_Detection.ipynb to generate the model metadata.")
    st.stop()

if not MODEL_PATH.is_file():
    st.error(f"The trained classifier was not found at `{MODEL_PATH}`.")
    st.info("Run all cells in RiceLeaf_Disease_Detection.ipynb to train and save the model.")
    st.stop()

try:
    model = load_classifier(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"Could not load the trained TensorFlow model: {error}")
    st.exception(error)
    st.stop()

upload_column, detail_column = st.columns([1.08, 0.92], gap="large")
uploaded_fingerprint = None

with upload_column:
    with st.container():
        st.markdown('<div class="section-kicker">01 · YOUR PHOTO</div>', unsafe_allow_html=True)
        st.subheader("Upload a leaf photo")
        uploaded_file = st.file_uploader(
            "Choose a clear image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="JPEG or PNG · maximum 10 MB",
        )
        if uploaded_file is not None:
            if uploaded_file.type not in ALLOWED_IMAGE_TYPES:
                st.error("Unsupported image format. Please upload a JPEG or PNG image.")
                uploaded_file = None
            elif uploaded_file.size > MAX_UPLOAD_BYTES:
                st.error("This image is larger than 10 MB. Choose a smaller image.")
                uploaded_file = None
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            uploaded_fingerprint = hashlib.sha256(image_bytes).hexdigest()
            try:
                rgb_image = decode_uploaded_image(image_bytes)
            except (ValueError, cv2.error) as error:
                st.error(str(error))
                rgb_image = None
            if rgb_image is not None:
                st.image(rgb_image, caption="Ready for analysis", use_column_width=True)
        else:
            image_bytes = None
            rgb_image = None
            st.markdown(
                '<div class="small-note">Use one clear, well-lit leaf photo. The model will compare it '
                'with the three classes learned during training.</div>',
                unsafe_allow_html=True,
            )
        analyze = st.button(
            "Analyze image  →",
            type="primary",
            use_container_width=True,
            disabled=rgb_image is None,
        )

if uploaded_fingerprint is None or st.session_state.get("scan_fingerprint") != uploaded_fingerprint:
    st.session_state.pop("scan_result", None)
    st.session_state.pop("scan_fingerprint", None)

with detail_column:
    if analyze and rgb_image is not None:
        st.session_state.pop("scan_result", None)
        st.session_state.pop("scan_fingerprint", None)
        with st.spinner("Looking at the leaf image..."):
            try:
                scores = predict_image(model, rgb_image, class_names)
            except (ValueError, cv2.error, tf.errors.OpError) as error:
                st.error(f"Image analysis failed: {error}")
                scores = None

        if scores is not None:
            best_index = int(np.argmax(scores))
            st.session_state["scan_fingerprint"] = uploaded_fingerprint
            st.session_state["scan_result"] = {
                "fingerprint": uploaded_fingerprint,
                "filename": uploaded_file.name,
                "predicted_class": class_names[best_index],
                "score": float(scores[best_index]),
                "scores": scores.astype(float).tolist(),
                "image_dimensions": f"{rgb_image.shape[1]} × {rgb_image.shape[0]} px",
            }

    scan_result = st.session_state.get("scan_result")
    if scan_result and scan_result.get("fingerprint") == uploaded_fingerprint:
        confidence = scan_result["score"]
        best_class = scan_result["predicted_class"]
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">02 · SCAN RESULT</div>
                <div class="result-name">{best_class}</div>
                <span class="confidence-pill">Model score · {confidence:.1%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("#### Class scores")
        score_frame = pd.DataFrame(
            {"Disease class": class_names, "Model score (%)": np.asarray(scan_result["scores"]) * 100}
        ).sort_values("Model score (%)", ascending=False)
        st.bar_chart(score_frame.set_index("Disease class"), height=220)
        st.info(
            "This score is the model's output, not a calibrated probability. "
            "The model learned from only 119 images and can be wrong, especially on field photos."
        )
    else:
        st.markdown('<div class="section-kicker">02 · READY FOR YOUR SCAN</div>', unsafe_allow_html=True)
        st.markdown("### Your scan results")
        st.info("Upload a leaf photo and select **Analyze image** to see its predicted class and scores here.")
        st.markdown(
            '<div class="small-note">The model compares the uploaded photo with bacterial leaf blight, '
            'brown spot, and leaf smut examples.</div>',
            unsafe_allow_html=True,
        )

st.write("")
st.markdown("---")
scan_result = st.session_state.get("scan_result")
if scan_result and scan_result.get("fingerprint") == uploaded_fingerprint:
    st.markdown("## Your scan")
    st.markdown(
        '<div class="small-note">This summary updates when you analyze a new photo.</div>',
        unsafe_allow_html=True,
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric("Image analyzed", scan_result["filename"])
    summary_columns[1].metric("Predicted class", scan_result["predicted_class"])
    summary_columns[2].metric("Model score", f"{scan_result['score']:.1%}")
    st.caption(f"Image dimensions: {scan_result['image_dimensions']}")
else:
    with st.expander("About the trained model"):
        st.caption("These describe the trained model; they are not results for your uploaded photo.")
        detail_columns = st.columns(3)
        detail_columns[0].metric("Dataset images", str(len(class_metadata)))
        detail_columns[1].metric("Disease classes", str(len(class_names)))
        if METRICS_PATH.is_file():
            comparison_metrics = pd.read_csv(METRICS_PATH)
            transfer_metrics = comparison_metrics[
                comparison_metrics["model"].astype(str).str.startswith("MobileNetV2")
            ]
            test_accuracy = float(transfer_metrics.iloc[0]["accuracy"]) if not transfer_metrics.empty else None
        else:
            test_accuracy = None
        detail_columns[2].metric(
            "Held-out test accuracy",
            f"{test_accuracy:.1%}" if test_accuracy is not None else "Not available",
            help="Measured on the project's small held-out test split; not a per-photo confidence score.",
        )

if METRICS_PATH.is_file():
    with st.expander("See model comparison from the project notebook"):
        try:
            comparison = pd.read_csv(METRICS_PATH)
            display_table = comparison.copy()
            for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                if metric in display_table:
                    display_table[metric] = display_table[metric].map(lambda value: f"{value:.1%}")
            st.dataframe(display_table, use_container_width=True)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Could not read the notebook's model comparison: {error}")

st.markdown(
    """
    <div class="small-note">
    The dataset archive contained 40 Bacterial leaf blight, 40 Brown spot, and 39 Leaf smut images.
    The model is a learning demonstration, not validated for field use. Seek expert advice for crop
    health decisions; do not use this result to choose treatments.
    </div>
    """,
    unsafe_allow_html=True,
)
