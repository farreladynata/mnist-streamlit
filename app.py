import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("🔢 MNIST Digit Recognition")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.h5")

try:
    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload gambar angka (png/jpg/jpeg)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Gambar diupload", width=200)

        image = image.resize((28, 28))
        img = np.array(image)

        img = 255 - img
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)
        digit = np.argmax(pred)
        conf = float(np.max(pred) * 100)

        st.success(f"Prediksi: {digit}")
        st.info(f"Confidence: {conf:.2f}%")

except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
