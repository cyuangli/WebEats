import streamlit as st
from pathlib import Path
from PIL import Image
import os

from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils import load_keras, load_joblib, load_faiss, load_npy
from src.config.configuration import ConfigurationManager

# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parent / "notebooks"

# ------------------ Page config ------------------
st.set_page_config(
    page_title="Recipe Image Similarity Search",
    layout="wide"
)

# ------------------ Dark grid styling ------------------
st.markdown(
    """
    <style>
    .image-grid {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 14px;
        margin-top: 20px;
    }
    .image-grid img {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Load heavy assets ONCE ------------------
@st.cache_resource
def load_pipeline():
    config = ConfigurationManager()
    model_dir = config.get_model_training_config().save_path

    embedding_model = load_keras(os.path.join(model_dir, "embedding_model.keras"))
    pca = load_joblib(os.path.join(model_dir, "pca.joblib"))
    index = load_faiss(os.path.join(model_dir, "recipes.faiss"))
    image_paths = load_npy(os.path.join(model_dir, "image_paths.npy"))

    return PredictionPipeline(
        embedding_model=embedding_model,
        pca=pca,
        index=index,
        image_paths=image_paths
    )

pipeline = load_pipeline()

# ------------------ App title ------------------
st.title("🍲 Recipe Image Similarity Search")

# ------------------ Upload image ------------------
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # ---- Show query image ----
    input_image = Image.open(uploaded_file)
    st.subheader("📷 Query Image")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(input_image, width=350)

    # ---- Similarity search ----
    with st.spinner("Finding similar recipes..."):
        image_paths = pipeline.initiate_pipeline(uploaded_file, k=25)

    # ---- Display results ----
    st.subheader("🔍 Most Similar Recipes")
    st.markdown('<div class="image-grid">', unsafe_allow_html=True)

    cols = st.columns(5, gap="small")
    for i, rel_path in enumerate(image_paths):
        img_path = (PROJECT_ROOT / rel_path).resolve()
        with cols[i % 5]:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning("Image not found")

    st.markdown('</div>', unsafe_allow_html=True)
