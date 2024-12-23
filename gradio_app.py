import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from Augmentation import Augmentation
from Transformation import Transformation
import matplotlib.pyplot as plt
import json
import os

MODE_DESCRIPTIONS = {
    "Augmentation": """
    <div style="background-color: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ecc71; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <h3 style="color: #2ecc71; margin-bottom: 10px;">About Data Augmentation</h3>
        <p style="color: #bdc3c7; line-height: 1.6;">
        Data augmentation increases training data diversity through various transformations.
        <br><br>
        This process improves model robustness and prevents overfitting by simulating different real-world conditions.
        <br><br>
        Each transformation method serves a specific purpose:
        <br><br>
        • Rotation: Helps model recognize leaves at different angles<br><br>
        • Blur: Simulates different image qualities and focus levels<br><br>
        • Flip: Creates mirror images to double the dataset<br><br>
        • Zoom: Handles different distance variations<br><br>
        • Contrast/Brightness: Adapts to various lighting conditions
        <br><br>
        Please select an image and apply different augmentation techniques to see the results.
        </p>
    </div>
    """,
    "Transform": """
    <div style="background-color: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ecc71; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <h3 style="color: #2ecc71; margin-bottom: 10px;">About Image Transformation</h3>
        <p style="color: #bdc3c7; line-height: 1.6;">
        Image transformation techniques are essential for preprocessing and feature extraction in computer vision.
        <br><br>
        These methods enhance specific aspects of leaf images for better disease detection:
        <br><br>
        • Gaussian Blur: Reduces noise and smooths image details<br><br>
        • Mask: Isolates leaf from background for focused analysis<br><br>
        • ROI: Identifies key areas of interest in the leaf<br><br>
        • Object Analysis: Examines leaf morphology and structure<br><br>
        • Pseudolandmarks: Detects significant points for shape analysis<br><br>
        • Histogram: Analyzes color distribution patterns
        <br><br>
        Select an image and apply different transformation techniques to observe the results.
        </p>
    </div>
    """,
    "Predict": """
    <div style="background-color: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ecc71; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <h3 style="color: #2ecc71; margin-bottom: 10px;">About Disease Prediction</h3>
        <p style="color: #bdc3c7; line-height: 1.6;">
        Our disease prediction system utilizes state-of-the-art deep learning models powered by TensorFlow.
        <br><br>
        The system includes specialized models for different plant species:
        <br><br>
        • Apple Model: Trained on extensive apple leaf pathology dataset<br><br>
        • Grape Model: Specialized for grape disease detection
        <br><br>
        Each model uses convolutional neural networks (CNNs) to analyze leaf patterns, colors, and textures for accurate disease classification.
        <br><br>
        The model accuracy is 95%
        <br><br>
        Select an image and choose the model type to predict the disease.
        </p>
    </div>
    """
}

# Función para cargar imágenes de ejemplo
@gr.on_exception
def load_example_images(directory='examples'):
    examples = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                examples.append(os.path.join(directory, file))
    return examples

# Función para augmentación
def apply_augmentation(image, aug_method, intensity):
    try:
        os.makedirs('temp', exist_ok=True)
        temp_path = "temp/input_aug.jpg"
        
        if isinstance(image, str):
            Image.open(image).save(temp_path)
        else:
            Image.fromarray(image).save(temp_path)
            
        aug = Augmentation("")
        img = aug.get_img(temp_path)
        
        if img is None:
            raise ValueError("Failed to load image")
            
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
            
        output_path = "temp/output_aug.jpg"
        aug.save_img(output_path)
        
        result = cv2.imread(output_path)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        raise gr.Error(f"Error in augmentation: {str(e)}")

# Función para transformación
def apply_transformation(image, trans_method):
    try:
        os.makedirs('temp', exist_ok=True)
        temp_path = "temp/input_trans.jpg"
        
        if isinstance(image, str):
            Image.open(image).save(temp_path)
        else:
            Image.fromarray(image).save(temp_path)
            
        trans = Transformation(temp_path)
        
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
            plt.savefig('temp/histogram.png')
            plt.close()
            result = cv2.imread('temp/histogram.png')
            
        if result is not None:
            cv2.imwrite("temp/output_trans.jpg", result)
            result = cv2.imread("temp/output_trans.jpg")
            if result is not None:
                return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return None
        
    except Exception as e:
        raise gr.Error(f"Error in transformation: {str(e)}")

# Función para predicción
def predict_disease(image, model_type):
    try:
        os.makedirs('temp', exist_ok=True)
        temp_path = "temp/input_pred.jpg"
        
        if isinstance(image, str):
            Image.open(image).save(temp_path)
        else:
            Image.fromarray(image).save(temp_path)
            
        model_path = f"utils/{model_type}_best_model.keras"
        json_path = f"utils/{model_type}_class_indices.json"
        
        if not os.path.exists(model_path) or not os.path.exists(json_path):
            raise FileNotFoundError(f"Model files not found")
            
        img = cv2.imread(temp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        model = tf.keras.models.load_model(model_path)
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
            
        img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        class_name = {v: k for k, v in class_indices.items()}[predicted_class]
        
        height, width = img.shape[:2]
        result = np.zeros((height + 60, width * 2, 3), dtype=np.uint8)
        result[0:height, 0:width] = img
        result[0:height, width:] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Predicted: {class_name}", (10, height + 40), 
                    font, 1, (255, 255, 255), 2)
                    
        probabilities = {k: float(predictions[0][v]) for k, v in class_indices.items()}
        prob_text = " | ".join([f"{k}: {v*100:.1f}%" for k, v in probabilities.items()])
        cv2.putText(result, prob_text, (10, height + 20), font, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite("temp/prediction_result.jpg", result)
        return cv2.imread("temp/prediction_result.jpg")
        
    except Exception as e:
        raise gr.Error(f"Error in prediction: {str(e)}")

# Interfaz principal
with gr.Blocks(title="Advanced Leaf Disease Analysis") as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem; background: linear-gradient(90deg, #1a1a1a, #2d2d2d); border-radius: 10px;">
            <h1 style="color: #2ecc71;">🍃 Advanced Leaf Disease Analysis</h1>
            <p style="font-size: 1.2rem; color: #95a5a6;">
                Powered by TensorFlow & Computer Vision | Deep Learning for Plant Pathology
            </p>
        </div>
    """)
    
    # Selector de modo y descripción
    mode = gr.Radio(
        choices=["Augmentation", "Transform", "Predict"],
        label="Analysis Mode",
        value="Augmentation"
    )
    description = gr.HTML(value=MODE_DESCRIPTIONS["Augmentation"])
    
    # Contenedor principal
    with gr.Row():
        # Columna de entrada
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input Image")
            gr.Examples(
                load_example_images(),
                inputs=input_image,
                label="Available Images"
            )
            
            # Controles de Augmentation
            with gr.Column(visible=True) as aug_controls:
                aug_method = gr.Radio(
                    choices=["Rotation", "Blur", "Flip", "Zoom", "Contrast", "Brightness"],
                    label="Augmentation Method",
                    value="Rotation"
                )
                intensity = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    label="Intensity",
                    info="Adjust the intensity of the augmentation"
                )
                aug_button = gr.Button("Apply Augmentation", variant="primary")
            
            # Controles de Transform
            with gr.Column(visible=False) as trans_controls:
                trans_method = gr.Radio(
                    choices=["Gaussian Blur", "Mask", "ROI", "Object Analysis", "Pseudolandmarks", "Histogram"],
                    label="Transform Method",
                    value="Gaussian Blur"
                )
                trans_button = gr.Button("Apply Transform", variant="primary")
            
            # Controles de Predict
            with gr.Column(visible=False) as pred_controls:
                model_type = gr.Radio(
                    choices=["Apple", "Grape"],
                    label="Model Type",
                    value="Apple"
                )
                pred_button = gr.Button("Make Prediction", variant="primary")
        
        # Columna de resultado
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Result")
    
    # Eventos
    mode.change(lambda x: MODE_DESCRIPTIONS[x], inputs=[mode], outputs=[description])
    
    def update_visibility(mode):
        return [
            mode == "Augmentation",  # aug_controls
            mode == "Transform",     # trans_controls
            mode == "Predict"        # pred_controls
        ]
    
    mode.change(
        update_visibility,
        inputs=[mode],
        outputs=[
            aug_controls,
            trans_controls,
            pred_controls,
        ]
    )
    
    aug_button.click(
        apply_augmentation,
        inputs=[input_image, aug_method, intensity],
        outputs=[output_image]
    )
    
    trans_button.click(
        apply_transformation,
        inputs=[input_image, trans_method],
        outputs=[output_image]
    )
    
    pred_button.click(
        predict_disease,
        inputs=[input_image, model_type],
        outputs=[output_image]
    )

if __name__ == "__main__":
    demo.launch()