"""
Campus Waste Segregation Assistant - Streamlit app (corrected)

IMPORTANT
---------
The saved model already contains its own pixel-rescaling layer
(0-255 -> -1..1, the MobileNetV2 way). So this app must feed RAW pixel
values (0-255). Do NOT call preprocess_input here: doing that scales the
image twice and the model then sees an almost flat image, which is why the
old version predicted "metal" for a plastic bottle.

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import keras
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# 1. PAGE CONFIGURATION (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Waste Segregation Assistant",
    page_icon="♻️",
    layout="centered",
)


# ============================================================
# 2. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent          # works from any working directory
MODEL_PATH = BASE_DIR / "waste_segregation_mobilenetv2.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = 224
DEFAULT_CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

HIGH_CONFIDENCE = 70.0
LOW_CONFIDENCE = 50.0


# ============================================================
# 3. DISPOSAL RECOMMENDATIONS
# ============================================================

DISPOSAL_GUIDE = {
    "cardboard": {
        "icon": "📦",
        "title": "Cardboard",
        "message": "Place it in the cardboard/paper recycling category.",
        "tip": "Keep cardboard reasonably clean and dry before recycling.",
    },
    "glass": {
        "icon": "🍾",
        "title": "Glass",
        "message": "Place it in the appropriate glass recycling collection.",
        "tip": "Handle broken glass carefully and follow your local recycling rules.",
    },
    "metal": {
        "icon": "🥫",
        "title": "Metal",
        "message": "Place it in the metal recycling category.",
        "tip": "Empty and clean containers when possible before recycling.",
    },
    "paper": {
        "icon": "📄",
        "title": "Paper",
        "message": "Place it in the paper recycling category.",
        "tip": "Keep paper clean and dry for better recycling.",
    },
    "plastic": {
        "icon": "♻️",
        "title": "Plastic",
        "message": "Place it in the appropriate plastic recycling category.",
        "tip": "Empty and rinse containers when applicable before disposal.",
    },
    "trash": {
        "icon": "🗑️",
        "title": "General Trash",
        "message": "Place it in the general/non-recyclable waste category.",
        "tip": "Check local waste-management rules for special materials.",
    },
}


# ============================================================
# 4. LOAD MODEL AND CLASS NAMES
# ============================================================

@st.cache_resource
def load_model():
    # compile=False: we only predict, so no optimizer/loss is needed
    return keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Put it in the same folder as app.py.")
    st.stop()

model = load_model()
CLASS_NAMES = load_class_names()


# ============================================================
# 5. IMAGE PREPARATION (identical to load_image() in the notebook)
# ============================================================

def prepare_image(source):
    """Return (224, 224, 3) uint8 array with RAW pixel values 0-255."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)                  # fix rotated phone photos
    if img.mode in ("RGBA", "LA", "P"):                 # transparent PNG -> white background
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ============================================================
# 6. HEADER
# ============================================================

st.title("♻️ Waste Segregation Assistant")

st.write(
    "Upload or capture an image of a waste item and the AI will "
    "identify its most likely waste category."
)

st.info(
    "💡 For better results, use a clear image with a single item "
    "fully visible and good lighting."
)


# ============================================================
# 7. IMAGE INPUT (upload or camera)
# ============================================================

tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png", "webp"],
    )

with tab_camera:
    camera_file = st.camera_input("Take a photo of the item")

source_file = uploaded_file if uploaded_file is not None else camera_file


# ============================================================
# 8. PROCESS IMAGE
# ============================================================

if source_file is not None:

    try:
        image_array = prepare_image(source_file)        # uint8, 0-255
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
        st.stop()

    st.image(image_array, caption="Image as seen by the model (224x224)", use_container_width=True)

    # Raw pixels, float32, plus batch dimension. NO preprocess_input here.
    batch = np.expand_dims(image_array.astype("float32"), axis=0)

    predictions = model.predict(batch, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]
    predicted_index = int(top_indices[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100
    runner_up = float(predictions[int(top_indices[1])]) * 100

    guide = DISPOSAL_GUIDE[predicted_class]

    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    st.divider()
    st.subheader("🔍 AI Prediction")

    if confidence >= LOW_CONFIDENCE:
        st.success(f"{guide['icon']} {guide['title']}")
    else:
        st.warning(f"🤔 Best guess: {guide['icon']} {guide['title']}")

    st.metric("Confidence", f"{confidence:.2f}%")

    if confidence >= HIGH_CONFIDENCE:
        st.success("✅ High confidence prediction")
    elif confidence >= LOW_CONFIDENCE:
        st.warning("⚠️ Moderate confidence prediction")
    else:
        st.error("❗ Low confidence prediction")
        st.write(
            "The model is not sure about this image. Try a clearer photo, "
            "a plain background, and make sure the item fills most of the frame."
        )

    if confidence - runner_up < 15:
        st.caption("The top two categories are very close, so double-check the item.")

    # --------------------------------------------------------
    # Top 3
    # --------------------------------------------------------

    st.subheader("📊 Top 3 Predictions")

    for rank, index in enumerate(top_indices, start=1):
        probability = float(predictions[int(index)])
        st.write(f"**{rank}. {CLASS_NAMES[int(index)].capitalize()}**")
        st.progress(probability, text=f"{probability * 100:.2f}%")

    # --------------------------------------------------------
    # Disposal recommendation
    # --------------------------------------------------------

    st.subheader("♻️ Recommended Action")
    st.write(guide["message"])
    st.caption(f"💡 {guide['tip']}")

    # --------------------------------------------------------
    # Debug info (helps catch preprocessing mistakes quickly)
    # --------------------------------------------------------

    with st.expander("🛠️ Debug info"):
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Array sent to model: shape={batch.shape}, dtype={batch.dtype}")
        st.write(f"Pixel range: min={batch.min():.0f}, max={batch.max():.0f}  (must be about 0 to 255)")
        st.write({CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(predictions)})


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()
st.caption("Waste Segregation Assistant • Powered by a fine-tuned MobileNetV2 model")
