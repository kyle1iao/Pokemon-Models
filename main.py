import streamlit as st

st.set_page_config(page_title="PokéScan", page_icon="🔍")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import time
import base64
import requests
from PIL import Image
from io import BytesIO


# Load model
@st.cache_resource()
def load_model():
    class CustomGlobalAveragePooling2D(Layer):
        def __init__(self, **kwargs):
            super(CustomGlobalAveragePooling2D, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.reduce_mean(inputs, axis=[1, 2])

    tf.keras.utils.get_custom_objects()[
        "CustomGlobalAveragePooling2D"
    ] = CustomGlobalAveragePooling2D

    return tf.keras.models.load_model("model_pokemon.h5")


model = load_model()


@st.cache_resource
def load_generator():
    data_generator = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = data_generator.flow_from_directory(
        "data/ImageSplit/train", target_size=(256, 256), class_mode="categorical"
    )
    return train_gen


# Function for predicting given an image array
def predict_pokemon(image_array):
    predictions = model.predict(image_array)
    train_gen = load_generator()
    class_labels = train_gen.class_indices
    class_labels = {v: k for k, v in class_labels.items()}
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label


def get_example(pokemon):
    pokemon = pokemon.lower()
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon}"
    r = requests.get(url)

    data = r.json()
    forms = data["forms"][0]["url"]

    form_link = requests.get(forms)
    content = form_link.json()
    image_url = content["sprites"]["front_default"]

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image


def main():
    LOGO_IMAGE = "sphealimage.png"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
            align-items: center;
        }
        .logo-text {
            font-family: "Source Sans Pro", sans-serif;
            font-weight: 650;
            color: rgb(250, 250, 250);
            font-size: calc(1.4rem + 3.0vw);
        }
        .logo-img {
            float:right;
            width: calc(1.4rem + 3.0vw);
            height: calc(1.4rem + 3.0vw);
            margin-left: 20px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="container">
            <p class="logo-text">PokéScan</p>
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # st.title("PokéScan")
    st.header("Ever wonder what Pokémon you would be?")

    uploaded_file = st.file_uploader(
        "Upload a selfie to find out!",
        type=["jpg", "png"],
        help="Supported file types include jpeg and png.",
    )

    if not uploaded_file:
        st.stop()

    with st.spinner(text="Identifying that Pokémon..."):
        bar = st.progress(0)
        time.sleep(0.5)
        bar.progress(25)
        time.sleep(0.5)
        bar.progress(50)
        time.sleep(0.5)
        bar.progress(100)

        image_display = load_img(uploaded_file)

        centered_container = st.container()

        with centered_container:
            st.image(image_display, caption="", use_column_width=True)

        # process image
        image = load_img(uploaded_file, target_size=(256, 256))
        image_array = img_to_array(image) / 255.0
        image_array = image_array.reshape(1, 256, 256, 3)

        # Make a prediction and display the result
        predicted_pokemon = predict_pokemon(image_array)
        st.write("## Predicted Pokémon:", predicted_pokemon)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("In-game pixel sprite of", predicted_pokemon, ":")
        with col2:
            st.image(get_example(predicted_pokemon), width=200)


if __name__ == "__main__":
    main()
