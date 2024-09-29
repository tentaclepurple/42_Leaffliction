import os
import sys
import shutil


def remove_empty_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        if not os.listdir(directory):  # Check if the directory is empty
            os.rmdir(directory)
            print(f"Empty directory '{directory}' has been removed.")
        else:
            print(f"Directory '{directory}' is not empty and was not removed.")
    else:
        print(f"Directory '{directory}' does not exist.")


def organize_directories(images_dir):
    parent_dir = os.path.dirname(images_dir)
    apple_dir = os.path.join(parent_dir, 'Apple')
    grape_dir = os.path.join(parent_dir, 'Grape')

    if not os.path.exists(apple_dir):
        os.makedirs(apple_dir)
    if not os.path.exists(grape_dir):
        os.makedirs(grape_dir)

    for item in os.listdir(images_dir):
        item_path = os.path.join(images_dir, item)
        if os.path.isdir(item_path):
            if item.lower().startswith('apple'):
                shutil.move(item_path, os.path.join(apple_dir, item))
            elif item.lower().startswith('grape'):
                shutil.move(item_path, os.path.join(grape_dir, item))

    #remove_empty_directory(images_dir)

    print("'Apple' and 'Grape' directories created")


def main():
    if len(sys.argv) != 2:
        print("Usage: python OrganizeDirectories.py <images_directory>")
        sys.exit(1)

    images_directory = sys.argv[1]
    organize_directories(images_directory)


if __name__ == "__main__":
    main()
