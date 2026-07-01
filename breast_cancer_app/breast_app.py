import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

CLASS_NAMES = ["benign", "malignant", "normal"]


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )
    return model


model = load_model()

st.title("🩺 Breast Cancer Detection using Deep Learning")
st.write("Upload a Breast Ultrasound Image for Classification.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)

    # IMPORTANT:
    # DO NOT call preprocess_input()
    # because your model already contains Lambda(preprocess_input)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    with col2:

        st.success(f"Prediction: **{predicted_class.upper()}**")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        for cls, score in zip(CLASS_NAMES, prediction[0]):
            st.write(f"**{cls.capitalize()}**")
            st.progress(float(score))
            st.write(f"{score*100:.2f}%")

    st.divider()

    st.subheader("Raw Output")

    st.write(prediction)