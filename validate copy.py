import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# Define the class names as per your dataset
CLASS_NAMES = ['Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']


def check_arguments():
    """
    Function to validate the arguments passed to the script.
    Ensures an image file is provided and valid.
    """
    if len(sys.argv) != 2:
        print("Usage: python validate.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.isfile(image_path):
        print(f"Error: {image_path} is not a valid file.")
        sys.exit(1)
    
    return image_path


def load_and_prepare_image(image_path):
    """
    Loads and preprocesses the image for prediction.
    Assumes images are already 250x250, converts them to array,
    and normalizes them to the same scale as during training.
    """
    print(f"Loading and preprocessing image: {image_path}")
    
    # Load the image (no resizing since they're already 250x250)
    img = image.load_img(image_path)
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    
    # Add an extra dimension (for batch size)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image (same normalization as during training: rescale=1./255)
    img_array /= 255.0
    
    return img_array


def load_model_and_predict(image_array, model_path='best_model.keras'):
    """
    Loads the trained model and uses it to predict the class of the input image.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print("Predicting the class of the image...")
    predictions = model.predict(image_array)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    
    return predicted_class_index, predictions[0]


def main():
    # Step 1: Validate and get the image path
    image_path = check_arguments()

    # Step 2: Load and preprocess the image
    image_array = load_and_prepare_image(image_path)

    # Step 3: Load the model and make a prediction
    predicted_class_index, predictions = load_model_and_predict(image_array)

    # Step 4: Output the results
    print(f"Predicted class: {CLASS_NAMES[predicted_class_index]}")
    print(f"Predictions (probabilities): {predictions}")


if __name__ == "__main__":
    main()
