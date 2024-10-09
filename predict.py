import sys
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt


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


def load_model_and_predict(image_array, model_path, class_indices):
    """
    Loads the trained model and uses it to predict
    the class of the input image.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Invert the class indices to map from index to class name
    class_labels = {v: k for k, v in class_indices.items()}

    print("Predicting the class of the image...")
    predictions = model.predict(image_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    return predicted_class_index, predictions[0], class_labels


def load_and_prepare_image(image_path):
    """
    Loads and preprocesses the image for prediction.
    Assumes the image is already 250x250.
    Converts the image to an array and normalizes
    it to the same scale as during training.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = image.img_to_array(img_rgb)

    img_array = np.expand_dims(img_array, axis=0)

    img_array /= 255.0

    return img_array


def extract_white_background(img):
    """
    Extracts the white background from the image
    and replaces it with pure white.
    Detects pixels that are close to white and
    replaces them with a white background.
    """

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Threshold on 'a' channel (green-red)
    _, a_thresh = cv2.threshold(a, 127, 255,
                                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    a_thresh = cv2.morphologyEx(a_thresh, cv2.MORPH_CLOSE, kernel)
    a_thresh = cv2.morphologyEx(a_thresh, cv2.MORPH_OPEN, kernel)

    # Apply Gaussian blur to smooth the edges
    gaussmask = cv2.GaussianBlur(a_thresh, (3, 3), 0)

    # Apply the mask
    masked_image = pcv.apply_mask(img=img, mask=gaussmask, mask_color="white")

    return masked_image


def create_composite_image(original_image, processed_image, class_name):
    """
    Creates a composite image with the original image
    and a processed version with white background.
    Adds a frame with "DL classification" and "Class predicted: ...".
    """
    # Create a black background for the entire frame
    frame_height = original_image.shape[0] + 100  # Extra space for text below
    frame_width = original_image.shape[1] * 2  # Twice the width to fit images
    black_background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    black_background[0:original_image.shape[0],
                     0:original_image.shape[1]] = original_image
    black_background[0:processed_image.shape[0],
                     original_image.shape[1]:] = processed_image

    # Create a matplotlib figure to add the text
    fig, ax = plt.subplots()
    ax.imshow(black_background)
    ax.axis('off')

    # Add "DL classification" label
    plt.text(
        black_background.shape[1] // 2, original_image.shape[0] + 20,
        "=== DL classification ===", fontsize=14, color="white",
        ha="center", backgroundcolor="black"
    )

    # Add predicted class text
    plt.text(
        black_background.shape[1] // 2, original_image.shape[0] + 50,
        f"Class predicted: {class_name}", fontsize=16,
        color="green", ha="center"
    )

    # Show the final image
    plt.show()


def main():
    image_path = check_arguments()

    species = None

    if 'apple' in image_path.lower():
        species = 'Apple'
    elif 'grape' in image_path.lower():
        species = 'Grape'
    else:
        print("Error: Model not found for the provided image.")
        sys.exit(1)

    model_path = input("Introduce the model file "
                       "you want to use ('.keras'): \n")
    model_path = f'utils/{model_path}'

    json_path = f'utils/{species}_class_indices.json'
    print(f"Loading class indices from {json_path}...")
    with open(json_path, 'r') as f:
        class_indices = json.load(f)

    original_image = cv2.imread(image_path)
    image_array = load_and_prepare_image(image_path)
    processed_image = extract_white_background(original_image)
    predicted_class_index, predictions, class_labels = load_model_and_predict(
        image_array, model_path, class_indices
        )

    class_name = class_labels[predicted_class_index]
    print("Predictions (probabilities) for each class:")
    for clas, prob in zip(class_indices.keys(), predictions):
        print(f"{clas}: {prob:.4f}")

    print(f"Predicted class: {class_name}")

    create_composite_image(original_image, processed_image, class_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
