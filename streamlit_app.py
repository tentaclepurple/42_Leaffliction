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
            "content": "Data augmentation increases training data diversity through various transformations. \n\n"
            "This process improves model robustness and prevents overfitting by simulating different real-world conditions. \n\n"
            "Each transformation method serves a specific purpose:\n\n"
            + "<br>" +
            "• Rotation: Helps model recognize leaves at different angles\n\n• Blur: Simulates different image qualities and focus levels\n\n• Flip: Creates mirror images to double the dataset\n\n• Zoom: Handles different distance variations\n\n• Contrast/Brightness: Adapts to various lighting conditions\n\n"
            + "<br>" +
            "Please select an image below and apply different augmentation techniques, choose the intensity level with the slider, and click the button to see the results."
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
            "Select an image below and apply different transformation techniques to observe the results."
        },
        "Predict": {
            "title": "About Disease Prediction",
            "content": "Our disease prediction system utilizes state-of-the-art deep learning models powered by TensorFlow. \n\nThe system includes specialized models for different plant species:\n\n"
            + "<br>" +
            "• Apple Model: Trained on extensive apple leaf pathology dataset\n\n• Grape Model: Specialized for grape disease detection\n\n"
            + "<br>" +
            "Each model uses convolutional neural networks (CNNs) to analyze leaf patterns, colors, and textures for accurate disease classification.\n\nThe model accuracy is 95%\n\n"
            + "<br>" +
            "Select an image below and choose the model type to predict the disease. Click the button to see the analysis results."
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
        os.makedirs('temp', exist_ok=True)
        
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load model and class indices
        model_path = f"utils/{model_type}_best_model.keras"
        json_path = f"utils/{model_type}_class_indices.json"
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, None
        
        if not os.path.exists(json_path):
            st.error(f"Class indices file not found: {json_path}")
            return None, None, None
        
        model = tf.keras.models.load_model(model_path)
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        
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
        
        # Create result image with text
        result = np.zeros((height + 60, width * 2, 3), dtype=np.uint8)
        result[0:height, 0:width] = img
        result[0:height, width:] = processed
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Predicted: {class_name}", (10, height + 40), 
                    font, 1, (255, 255, 255), 2)
        
        output_path = 'temp/prediction_result.jpg'
        cv2.imwrite(output_path, result)
        
        return output_path, class_name, {k: float(predictions[0][v]) for k, v in class_indices.items()}
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None

def main():
    st.set_page_config(
        page_title="Leaf Disease Analysis | TensorFlow AI",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    # Sidebar - Simple mode selection
    with st.sidebar:
        st.markdown('<div class="section-title">Analysis Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Select Analysis Mode", ["Augmentation", "Transform", "Predict"])
    
    # Main title
    st.markdown(
        """
        <div class="main-title">
            <h1>🍃 Advanced Leaf Disease Analysis</h1>
            <p style="font-size: 1.2rem; color: #95a5a6;">
            Powered by TensorFlow & Computer Vision | Deep Learning for Plant Pathology
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Mode description
    mode_info = get_mode_description(mode)
    st.markdown(
        f"""
        <div class="info-box">
            <h3>{mode_info["title"]}</h3>
            <p>{mode_info["content"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Main content
    examples = load_example_images()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
        selected_image = st.selectbox(
            "Select an image for analysis",
            [img["name"] for img in examples]
        )
        
        if selected_image:
            image_path = next(img["path"] for img in examples if img["name"] == selected_image)
            st.image(image_path, caption="Selected Image", use_container_width=True)
            
            if mode == "Augmentation":
                aug_method = st.radio(
                    "Select Augmentation Method",
                    ["Rotation", "Blur", "Flip", "Zoom", "Contrast", "Brightness"]
                )
                intensity = st.slider(
                    "Adjust Intensity",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="Adjust the intensity of the augmentation effect"
                )
                
                if st.button("Apply Augmentation", type="primary"):
                    with st.spinner("Applying augmentation..."):
                        result_path = apply_augmentation(image_path, aug_method, intensity)
                        if result_path:
                            with col2:
                                st.markdown('<div class="section-title">Result</div>' + '<br><br><br>', unsafe_allow_html=True)
                                st.image(result_path, caption=f"{aug_method} Result", use_container_width=True)
            
            elif mode == "Transform":
                trans_method = st.radio(
                    "Select Transformation Method",
                    ["Gaussian Blur", "Mask", "ROI", "Object Analysis", "Pseudolandmarks", "Histogram"]
                )
                
                if st.button("Apply Transform", type="primary"):
                    with st.spinner("Applying transformation..."):
                        result_path = apply_transformation(image_path, trans_method)
                        if result_path:
                            with col2:
                                st.markdown('<div class="section-title">Result/div>', unsafe_allow_html=True)
                                st.image(result_path, caption=f"{trans_method} Result", use_container_width=True)
            
            elif mode == "Predict":
                model_type = st.radio(
                    "Select Model Type",
                    ["Apple", "Grape"]
                )
                
                if st.button("Make Prediction", type="primary"):
                    with st.spinner("Analyzing image..."):
                        result_path, class_name, probabilities = predict_disease(image_path, model_type)
                        if result_path:
                            with col2:
                                st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
                                st.image(result_path, caption="Disease Analysis", use_container_width=True)
                                
                                st.markdown("### Confidence Levels")
                                for class_name, prob in probabilities.items():
                                    st.progress(prob)
                                    st.markdown(f"**{class_name}**: {prob*100:.1f}%")

if __name__ == "__main__":
    main()