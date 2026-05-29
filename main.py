
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

st.set_page_config(
    page_title="StyleSnap AI",
    page_icon="👗",
    layout="wide"
)
import base64

def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_base64("images/fashion_bg.png")

page_bg = f"""
<style>

[data-testid="stAppViewContainer"] {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

h1 {{
text-align:center;
font-size:55px !important;
font-weight:800 !important;
color:white !important;
text-shadow:0px 0px 20px rgba(255,255,255,0.6);
}}

[data-testid="stFileUploader"] {{
background: rgba(20,20,30,0.42);
padding: 28px;
border-radius: 28px;
backdrop-filter: blur(18px);
max-width: 760px;
margin: auto;
box-shadow: 0px 12px 35px rgba(0,0,0,0.35);
border: 1px solid rgba(255,255,255,0.12);
transition: all 0.3s ease;
}}

[data-testid="stFileUploader"]:hover {{
transform: translateY(-4px);
box-shadow: 0px 15px 40px rgba(255,105,180,0.18);
border: 1px solid rgba(255,192,203,0.2);
}}
[data-testid="stImage"] img {{
border-radius:20px;
box-shadow:0px 8px 25px rgba(0,0,0,0.5);
transition:0.3s;
}}

[data-testid="stImage"] img:hover {{
transform:scale(1.08);
}}
[data-testid="stAppViewContainer"]::before {{
content: "";
position: fixed;
top: 0;
left: 0;
right: 0;
bottom: 0;
background: rgba(0,0,0,0.35);
pointer-events: none;
z-index: 0;
}}

[data-testid="stAppViewContainer"] > .main {{
position: relative;
z-index: 1;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

@st.cache_resource
def load_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )
    base_model.trainable = False

    model = tensorflow.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    return model

model = load_model()

st.markdown("""
<h1 style='text-align:center;
font-size:80px;
font-weight:900;
margin-bottom:0;
color:white;
text-shadow:0px 0px 25px rgba(255,255,255,0.7);'>
✨ StyleSnap AI
</h1>

<p style='text-align:center;
font-size:26px;
color:#f2f2f2;
margin-top:-10px;
margin-bottom:50px;
font-weight:500;'>
Discover outfits that match your aesthetic instantly 👗
</p>
""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;
font-size:18px;
color:white;
margin-bottom:25px;'>
──────── ✦ Find Your Perfect Match ✦ ────────
</div>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):

    try:
        os.makedirs("uploads", exist_ok=True)
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
st.markdown("""
<h3 style='text-align:center;color:white'>
📸 Upload Your Fashion Item
</h3>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    uploaded_file = st.file_uploader(
        "",
        type=['png','jpg','jpeg']
    )
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):

        # display the file
        display_image = Image.open(uploaded_file)
        st.markdown("""
        <h3 style='color:white;text-align:center'>
        ✨ Uploaded Fashion Item
        </h3>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(display_image, width=400)
        # feature extract
        features = feature_extraction(
            os.path.join("uploads", uploaded_file.name),
            model
        )

        indices = recommend(features, feature_list)
        # recommendention
        st.markdown("""
        <h2 style='text-align:center;
        color:white;
        font-size:42px;
        margin-top:50px;
        text-shadow:0px 0px 12px rgba(255,255,255,0.5);'>
        🔥 AI Picks Just For You
        </h2>

        <p style='text-align:center;
        color:#f5f5f5;
        font-size:18px;
        margin-top:-10px;
        margin-bottom:30px;'>
        Fashion pieces similar to your uploaded vibe ✨
        </p>
        """, unsafe_allow_html=True)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(
                "<h4 style='text-align:center;color:white;'>✨ Match 1</h4>",
                unsafe_allow_html=True
            )
            st.image(filenames[indices[0][0]], use_container_width=True)

        with col2:
            st.markdown(
                "<h4 style='text-align:center;color:white;'>💎 Match 2</h4>",
                unsafe_allow_html=True
            )
            st.image(filenames[indices[0][1]], use_container_width=True)

        with col3:
            st.markdown(
                "<h4 style='text-align:center;color:white;'>🔥 Match 3</h4>",
                unsafe_allow_html=True
            )
            st.image(filenames[indices[0][2]], use_container_width=True)

        with col4:
            st.markdown(
                "<h4 style='text-align:center;color:white;'>👗 Match 4</h4>",
                unsafe_allow_html=True
            )
            st.image(filenames[indices[0][3]], use_container_width=True)

        with col5:
            st.markdown(
                "<h4 style='text-align:center;color:white;'>🌟 Match 5</h4>",
                unsafe_allow_html=True
            )
            st.image(filenames[indices[0][4]], use_container_width=True)
