import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.h5")

st.title("🔢 MNIST Digit Recognition")
st.write("Upload gambar angka tulisan tangan (0-9)")

try:
    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload gambar",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        st.image(
            image,
            caption="Gambar yang diupload",
            width=200
        )

        image = image.resize((28, 28))

        img = np.array(image)

        img = 255 - img
        img = img / 255.0

        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Prediksi Angka: {digit}")
        st.info(f"Confidence: {confidence:.2f}%")

except Exception as e:
    st.error(f"Error loading model: {e}")
