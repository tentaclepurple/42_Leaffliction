import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import json
import os
import matplotlib.pyplot as plt
from Augmentation import Augmentation
from Transformation import Transformation

def load_css():
    st.markdown("""
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 200px;
            max-width: 600px;
            background-color: #1A1A1A;
        }
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
            color: #2ecc71;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #2ecc71;
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
        .info-box h3 { margin: 0 0 10px 0; color: #2ecc71; }
        .info-box p {
            margin: 0;
            line-height: 1.6;
            color: #bdc3c7;
            white-space: normal;
        }
        .method-description {
            font-size: 0.9rem;
            color: #bdc3c7;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: rgba(52, 152, 219, 0.1);
            border-radius: 4px;
        }
        .result-container {
            background-color: #2d2d2d;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)


def get_mode_description(mode):
    descriptions = {
        "Augmentation": {
            "title": "About Data Augmentation",
            "content": "Data augmentation increases training data diversity through various transformations. \n\n"
            "This process improves model robustness and prevents overfitting by simulating different real-world conditions. \n\n"
            "Each transformation method serves a specific purpose:\n\n"
            + "<br>" +
            "• Rotation: Helps model recognize leaves at different angles\n\n• Blur: Simulates different image qualities and focus levels\n\n• Flip: Creates mirror images to double the dataset\n\n• Zoom: Handles different distance variations\n\n• Contrast/Brightness: Adapts to various lighting conditions\n\n"
            + "<br>" +
            "Please select an image in the right and apply different augmentation techniques, choose the intensity level with the slider, and click the button to see the results."
        },
        "Transform": {
            "title": "About Image Transformation",
            "content": "Image transformation techniques are essential for preprocessing and feature extraction in computer vision.\n\n"
            "These methods enhance specific aspects of leaf images for better disease detection:\n\n"
            + "<br>" + 
            "• Gaussian Blur: Reduces noise and smooths image details\n\n"
            "• Mask: Isolates leaf from background for focused analysis\n\n"
            "• ROI: Identifies key areas of interest in the leaf\n\n"
            "• Object Analysis: Examines leaf morphology and structure\n\n"
            "• Pseudolandmarks: Detects significant points for shape analysis\n\n"
            "• Histogram: Analyzes color distribution patterns\n\n"
            + "<br>" +
            "Select an image in the right and apply different transformation techniques to observe the results."
        },
        "Predict": {
            "title": "About Disease Prediction",
            "content": "Our disease prediction system utilizes state-of-the-art deep learning models powered by TensorFlow. \n\nThe system includes specialized models for different plant species:\n\n"
            + "<br>" +
            "• Apple Model: Trained on extensive apple leaf pathology dataset\n\n• Grape Model: Specialized for grape disease detection\n\n"
            + "<br>" +
            "Each model uses convolutional neural networks (CNNs) to analyze leaf patterns, colors, and textures for accurate disease classification.\n\nThe model accuracy is 95%\n\n"
            + "<br>" +
            "Select an image in the right and choose the model type to predict the disease. Click the button to see the analysis results."
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

@st.cache_resource
def load_model_from_hub(model_type, repo_id):
    """Carga el modelo y los índices desde Hugging Face Hub"""
    try:
        model_filename = f"{model_type}_best_model.keras"
        indices_filename = f"{model_type}_class_indices.json"
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            repo_type="model"
        )
        
        indices_path = hf_hub_download(
            repo_id=repo_id,
            filename=indices_filename,
            repo_type="model"
        )
        
        model = tf.keras.models.load_model(model_path)
        with open(indices_path, 'r') as f:
            class_indices = json.load(f)
            
        return model, class_indices
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

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

def apply_transformation(image, trans_method):
    try:
        os.makedirs('temp', exist_ok=True)
        input_path = 'temp/input_trans.jpg'
        
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(input_path)
        elif isinstance(image, (str, bytes)):
            Image.open(image).save(input_path)
        
        trans = Transformation(input_path)
        
        if trans_method == "Gaussian Blur":
            result = trans.gaussian_blur()
        elif trans_method == "Mask":
            result = trans.create_mask()
        elif trans_method == "ROI":
            result = trans.roi_objects()
        elif trans_method == "Object Analysis":
            result = trans.analyze_object()
        elif trans_method == "Pseudolandmarks":
            result = trans.pseudolandmarks()
        elif trans_method == "Histogram":
            fig = trans.color_histogram()
            output_path = 'temp/histogram.png'
            plt.savefig(output_path)
            plt.close()
            return output_path
        
        output_path = 'temp/output_trans.jpg'
        cv2.imwrite(output_path, result)
        return output_path
    
    except Exception as e:
        st.error(f"Error in transformation: {str(e)}")
        return None

def predict_disease(image_path, model_type):
    try:
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cargar modelo e índices desde Hugging Face
        repo_id = "wolfframio/leaffliction"  # Reemplaza con tu repo
        model, class_indices = load_model_from_hub(model_type, repo_id)
        
        if model is None or class_indices is None:
            return None, None, None
        
        # Prepare image
        img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        class_name = {v: k for k, v in class_indices.items()}[predicted_class]
        
        # Create result visualization
        processed = cv2.imread(image_path)
        height, width = processed.shape[:2]
        
        result = np.zeros((height + 60, width * 2, 3), dtype=np.uint8)
        result[0:height, 0:width] = img
        result[0:height, width:] = processed
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Predicted: {class_name}", (10, height + 40), 
                    font, 1, (255, 255, 255), 2)
        
        output_path = 'temp/prediction_result.jpg'
        cv2.imwrite(output_path, result)
        
        return output_path, class_name, {k: float(predictions[0][v]) for k, v in class_indices.items()}
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None

def predict_disease(image_path, model_type):
    try:
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cargar modelo e índices desde Hugging Face
        repo_id = "tu-usuario/leaf-disease-models"  # Reemplaza con tu repo
        model, class_indices = load_model_from_hub(model_type, repo_id)
        
        if model is None or class_indices is None:
            return None, None, None
        
        # Prepare image
        img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        class_name = {v: k for k, v in class_indices.items()}[predicted_class]
        
        # Create result visualization
        processed = cv2.imread(image_path)
        height, width = processed.shape[:2]
        
        result = np.zeros((height + 60, width * 2, 3), dtype=np.uint8)
        result[0:height, 0:width] = img
        result[0:height, width:] = processed
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Predicted: {class_name}", (10, height + 40), 
                    font, 1, (255, 255, 255), 2)
        
        output_path = 'temp/prediction_result.jpg'
        cv2.imwrite(output_path, result)
        
        return output_path, class_name, {k: float(predictions[0][v]) for k, v in class_indices.items()}
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None