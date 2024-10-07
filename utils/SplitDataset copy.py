import os
import shutil
import random
import sys
from tqdm import tqdm


IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG']


def check_arguments():
    """
    Function to validate the arguments
    passed to the script
    """
    if len(sys.argv) != 2:
        print("Usage: python train.py <directory>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory.")
        sys.exit(1)
    
    return base_dir


def create_subdirectories(base_dir):
    """
    Function to create the train and validation
    folders replicating the subdirectory structure
    """
    train_dir = f"{base_dir}_train"
    validation_dir = f"{base_dir}_validation"

    # Create training and validation directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Iterate over subdirectories in the base directory and create corresponding folders
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, class_folder), exist_ok=True)

    return train_dir, validation_dir


def is_image_file(filename):
    """
    Check if a file is a valid
    image by extension
    """
    ext = os.path.splitext(filename)[1]
    return ext in IMAGE_EXTENSIONS


def split_images(base_dir, train_dir, validation_dir, split_ratio=0.8):
    """
    Function to split images into
    training and validation folders
    """

    expected_train_count = None
    expected_val_count = None

    for class_folder in tqdm(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_folder)
        
        if os.path.isdir(class_path):
            # List all images in the class
            images = os.listdir(class_path)
            random.shuffle(images)
            
            # Split the images according to the split_ratio
            split_point = int(len(images) * split_ratio)
            train_images = images[:split_point]
            validation_images = images[split_point:]

            # Set the expected counts if not set yet
            if expected_train_count is None:
                expected_train_count = len(train_images)
            if expected_val_count is None:
                expected_val_count = len(validation_images)

            # Check if the current class has the same number of images as expected
            if len(train_images) != expected_train_count:
                print(f"Error: The number of training images in "
                      "'{class_folder}' doesn't match the expected "
					  "count ({expected_train_count} images).")
                sys.exit(1)

            if len(validation_images) != expected_val_count:
                print(f"Error: The number of validation images in "
                      "'{class_folder}' doesn't match the expected count "
                      "({expected_val_count} images).")
                sys.exit(1)

            # Move the images to the corresponding folders
            for img in train_images:
                # Check if the file name contains 'histogram' and skip it if true
                if 'histogram' in img:
                    print(f"Skipping file {img} because it contains 'histogram'")
                    
                    continue  # Skip this file and do not move it """

                # Check if the file is a valid image
                if not is_image_file(img):
                    # print(f"Skipping {img}: not a valid image file")
                    continue  # Skip files that are not images

                shutil.move(os.path.join(class_path, img),
                            os.path.join(train_dir, class_folder, img))
            
            for img in validation_images:
                # Check if the file name contains 'histogram' and skip it if true
                if 'histogram' in img:
                    print(f"Skipping file {img} because it contains 'histogram'")
                    continue  # Skip this file and do not move it

                # Check if the file is a valid image
                if not is_image_file(img):
                    print(f"Skipping {img}: not a valid image file")
                    continue

                shutil.move(os.path.join(class_path, img),
                            os.path.join(validation_dir, class_folder, img))
