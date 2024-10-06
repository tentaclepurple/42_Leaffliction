import os
import sys
from utils import split_images, create_subdirectories


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


def main():
    # Check the arguments
    base_dir = check_arguments()
    
    # Create the training and validation folders
    train_dir, validation_dir = create_subdirectories(base_dir)
    
    # Split the images between training and validation
    split_images(base_dir, train_dir, validation_dir)

    print(f"Images successfully split into {train_dir} and {validation_dir}.")


if __name__ == "__main__":
    main()
