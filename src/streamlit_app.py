import streamlit as st
from PIL import Image
import tempfile
import os
from ml.infer import load_model, predict_with_gradcam

# -----------------------------------------------------------
#              STREAMLIT PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------
#              LOAD MODEL (cached for speed)
# -----------------------------------------------------------
@st.cache_resource
def load():
    return load_model()

model, classes = load()


# -----------------------------------------------------------
#              SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")
choice = st.sidebar.radio(
    "Go to:",
    ["📤 Upload Image", "📷 Webcam Capture", "ℹ️ About App"],
)

st.sidebar.markdown("---")
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

# Apply theme (simple simulation)
if theme == "Light":
    st.markdown(
        """
        <style>
        body { background-color: #f8f9fa !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------
#     FUNCTION: PROCESS IMAGE + SHOW RESULTS BLOCK
# -----------------------------------------------------------
def process_and_display(image_path, image_display):

    st.subheader("🌿 Uploaded Leaf")
    st.image(image_display, width=350)

    with st.spinner("Analyzing disease..."):
        label, confidence, gradcam_path = predict_with_gradcam(
            model=model, classes=classes, image_path=image_path
        )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.success(f"🩺 **Disease:** {label}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")

    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        gradcam_img = Image.open(gradcam_path)

        # Transparent overlay
        overlay = Image.blend(image_display.resize(gradcam_img.size), gradcam_img, alpha=0.45)
        st.image(overlay, caption="Grad-CAM Visualization")


# -----------------------------------------------------------
#                PAGE 1: FILE UPLOAD
# -----------------------------------------------------------
if choice == "📤 Upload Image":
    st.title("📤 Upload Leaf Image")

    uploaded = st.file_uploader("Upload a clear leaf image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            temp_path = tmp.name

        process_and_display(temp_path, img)


# -----------------------------------------------------------
#                PAGE 2: WEBCAM CAPTURE
# -----------------------------------------------------------
elif choice == "📷 Webcam Capture":
    st.title("📷 Capture Leaf Using Webcam")

    picture = st.camera_input("Take a photo")

    if picture:
        img = Image.open(picture).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            temp_path = tmp.name

        process_and_display(temp_path, img)


# -----------------------------------------------------------
#                 PAGE 3: ABOUT APP
# -----------------------------------------------------------
elif choice == "ℹ️ About App":
    st.title("ℹ️ About")
    st.write("""
    This Plant Disease Detection App uses **Deep Learning (ResNet)**  
    along with **Grad-CAM** explainability to highlight the diseased area.

    **Features:**
    - Upload image or use Webcam  
    - Grad-CAM heatmap overlay  
    - Fast GPU-optimized inference  
    - Clean modern UI with sidebar navigation  

    Developed for academic & real-world agricultural use. 🌱
    """)

    st.success("Made with ❤️ using Streamlit & PyTorch.")
