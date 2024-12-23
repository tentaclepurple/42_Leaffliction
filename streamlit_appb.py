import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from Augmentation import Augmentation
from Transformation import Transformation
import matplotlib.pyplot as plt
import json
import os

def load_css():
    st.markdown("""
        <style>
        .main-title {
            font-family: 'Helvetica Neue', sans-serif;
            color: #2ecc71;
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #1a1a1a, #2d2d2d);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .section-title {
            color: #3498db;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        
        .info-box {
            background-color: rgba(46, 204, 113, 0.1);
            border-left: 4px solid #2ecc71;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            max-height: none;
            overflow: visible;
        }
        
        .info-box h3 {
            margin: 0 0 10px 0;
            color: #2ecc71;
        }
        
        .info-box p {
            margin: 0;
            line-height: 1.6;
            color: #bdc3c7;
            white-space: normal;
        }
        
        .result-container {
            background-color: #2d2d2d;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def get_mode_description(mode):
    descriptions = {
        "Augmentation": {
            "title": "About Data Augmentation",
            "content": "Data augmentation increases training data diversity through various transformations. Select an image below and apply different augmentation techniques."
        },
        "Transform": {
            "title": "About Image Transformation",
            "content": "Image transformation techniques are essential for preprocessing and feature extraction. Select an image below and apply different transformation techniques."
        },
        "Predict": {
            "title": "About Disease Prediction",
            "content": "Our disease prediction system utilizes state-of-the-art deep learning models. Select an image and choose the model type to predict the disease."
        }
    }
    return descriptions[mode]

@st.cache_data
def load_example_images(directory='examples'):
    examples = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                examples.append({"name": file, "path": os.path.join(directory, file)})
    return examples


# Mostrar galería de imágenes
def display_image_gallery(images):
    st.markdown('### Input Image')  # Título
    selected_image_path = None

    # Crear columnas dinámicas para mostrar miniaturas pequeñas
    cols = st.columns(6)  # Se ajusta el tamaño con más columnas
    for idx, image in enumerate(images):
        col = cols[idx % 6]
        with col:
            if st.button("", key=f"btn_{image['name']}"):
                selected_image_path = image["path"]
            st.image(
                image["path"],
                caption=None,
                use_container_width=True  # Ajuste automático de tamaño
            )
    return selected_image_path


# Aplicar aumentación
def apply_augmentation(image, aug_method, intensity):
    try:
        os.makedirs('temp', exist_ok=True)
        input_path = 'temp/input_aug.jpg'

        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(input_path)
        elif isinstance(image, (str, bytes)):
            Image.open(image).save(input_path)

        aug = Augmentation("")
        img = aug.get_img(input_path)

        if img is None:
            st.error("Failed to load image")
            return None

        method_name = aug_method.lower()
        if method_name == "rotation":
            aug.rotate(intensity)
        elif method_name == "blur":
            aug.blur(intensity)
        elif method_name == "flip":
            aug.flip(intensity)
        elif method_name == "zoom":
            aug.zoom(intensity)
        elif method_name == "contrast":
            aug.add_contrast(intensity)
        elif method_name == "brightness":
            aug.add_brightness(intensity)

        output_path = 'temp/output_aug.jpg'
        aug.save_img(output_path)
        return output_path

    except Exception as e:
        st.error(f"Error in augmentation: {str(e)}")
        return None


# Aplicar transformación (ejemplo de funcionalidad adicional)
def apply_transformation(image, transform_method):
    try:
        os.makedirs('temp', exist_ok=True)
        input_path = 'temp/input_trans.jpg'

        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(input_path)
        elif isinstance(image, (str, bytes)):
            Image.open(image).save(input_path)

        trans = Transformation("")
        img = trans.get_img(input_path)

        if img is None:
            st.error("Failed to load image")
            return None

        method_name = transform_method.lower()
        if method_name == "grayscale":
            img = trans.to_grayscale()
        elif method_name == "edge_detection":
            img = trans.edge_detection()
        elif method_name == "thresholding":
            img = trans.thresholding()

        output_path = 'temp/output_trans.jpg'
        Image.fromarray(img).save(output_path)
        return output_path

    except Exception as e:
        st.error(f"Error in transformation: {str(e)}")
        return None


# Configuración principal de la aplicación
def main():
    st.set_page_config(
        page_title="Leaf Disease Analysis | TensorFlow AI",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Barra lateral
    with st.sidebar:
        st.markdown('### Analysis Mode')  # Título original
        mode = st.radio("Select Analysis Mode", ["Augmentation", "Transform", "Predict"])

    st.title("Advanced Leaf Disease Analysis")  # Título original

    # Cargar imágenes de ejemplo
    examples = load_example_images()
    with st.container():
        selected_image_path = display_image_gallery(examples)

        if selected_image_path:
            st.image(selected_image_path, caption="Selected Image", use_container_width=True)

            if mode == "Augmentation":
                aug_method = st.radio(
                    "Select Augmentation Method",
                    ["Rotation", "Blur", "Flip", "Zoom", "Contrast", "Brightness"]
                )
                intensity = st.slider("Adjust Intensity", min_value=0, max_value=100, value=50)

                if st.button("Apply Augmentation"):
                    result_path = apply_augmentation(selected_image_path, aug_method, intensity)
                    if result_path:
                        st.image(result_path, caption=f"{aug_method} Result", use_container_width=True)

            elif mode == "Transform":
                transform_method = st.radio(
                    "Select Transformation Method",
                    ["Grayscale", "Edge Detection", "Thresholding"]
                )

                if st.button("Apply Transformation"):
                    result_path = apply_transformation(selected_image_path, transform_method)
                    if result_path:
                        st.image(result_path, caption=f"{transform_method} Result", use_container_width=True)

            elif mode == "Predict":
                st.info("Prediction mode is not implemented yet. Stay tuned!")


if __name__ == "__main__":
    main()