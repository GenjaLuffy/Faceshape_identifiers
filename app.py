import streamlit as st
st.title("Face Shape Identifier")

import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gdown

if not os.path.exists("FaceShape.keras"):
    gdown.download(
        "YOUR_GOOGLE_DRIVE_FILE_LINK",
        "FaceShape.keras",
        quiet=False
    )

model = keras.models.load_model("FaceShape.keras")

path = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if path is not None:
    st.image(path)

    if st.button("Predict"):
        test_image = load_img(path, target_size=(224, 224,3))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # If you normalized images during training, uncomment this line:
        test_image = test_image / 255.0

        result = model.predict(test_image)

        classes = [
            "Heart",
            "Oblong",
            "Oval",
            "Round",
            "Square"
        ]

        predicted_class = np.argmax(result, axis=1)[0]
        output = classes[predicted_class]

        st.success(f"Face Shape: {output}")
        st.image(path) 

# Kaggle Dataset and Trainig Data code.
# https://www.kaggle.com/code/pawonshrestha/faceshape-using-vgg16/edit