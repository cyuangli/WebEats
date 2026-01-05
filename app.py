import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
from src.pipeline.predict_pipeline import PredictionPipeline

# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parent / "notebooks"
RECIPE_CSV = PROJECT_ROOT / "data/recipe_meta_topics.csv"

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
        border-radius: 10px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Load recipe metadata ------------------
@st.cache_data
def load_recipe_metadata():
    df = pd.read_csv(RECIPE_CSV)
    df["lemmatized_name"] = df["lemmatized_name"].str.strip().str.lower()
    return df

recipe_df = load_recipe_metadata()

# ------------------ Load pipeline once ------------------
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

# ------------------ Cache thumbnails ------------------
@st.cache_data
def load_thumbnail(img_path, size=(224, 224)):
    img = Image.open(img_path)
    img.thumbnail(size)
    return img

@st.cache_data
def load_full_image(img_path):
    return Image.open(img_path)

# ------------------ Helpers ------------------
def filename_to_lemmatized_name(image_path):
    stem = Path(image_path).stem
    stem = "_".join(stem.split("_")[:-1])
    return stem.replace("_", " ").lower()

# ------------------ Session state ------------------
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# ------------------ App title ------------------
st.title("🍲 Recipe Image Similarity Search")

# ------------------ Upload image ------------------
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # ---- Show query image (smaller) ----
    input_image = Image.open(uploaded_file)
    st.subheader("📷 Query Image")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(input_image, width=350)

    # ---- Run similarity search ----
    top_k = 25
    with st.spinner("Finding similar recipes..."):
        image_paths = pipeline.initiate_pipeline(uploaded_file, k=top_k)

    # ---- Display grid (use cached thumbnails) ----
    st.subheader("🔍 Most Similar Recipes")
    st.markdown('<div class="image-grid">', unsafe_allow_html=True)
    cols = st.columns(5, gap="small")

    for i, rel_path in enumerate(image_paths[:top_k]):
        img_path = (PROJECT_ROOT / rel_path).resolve()
        with cols[i % 5]:
            if img_path.exists():
                thumb = load_thumbnail(img_path)  # cached
                if st.button(" ", key=f"img_{i}"):
                    st.session_state.selected_image = rel_path
                st.image(thumb, use_container_width=True)
            else:
                st.warning("Image not found")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Recipe detail panel ------------------
if st.session_state.selected_image is not None:
    st.markdown("---")
    st.subheader("🍽️ Recipe Details")

    selected_path = st.session_state.selected_image
    lemma = filename_to_lemmatized_name(selected_path)
    match = recipe_df[recipe_df["lemmatized_name"] == lemma]

    if not match.empty:
        recipe_name = match.iloc[0]["original_name"]
        recipe_text = match.iloc[0]["recipe"]

        col_img, col_text = st.columns([1, 2])

        with col_img:
            img = load_full_image((PROJECT_ROOT / selected_path).resolve())
            st.image(img, use_container_width=True)

        with col_text:
            st.markdown(f"## {recipe_name}")
            steps = recipe_text.split("|")
            for step in steps:
                st.markdown(f"- {step.strip()}")
    else:
        st.warning("Recipe metadata not found.")
