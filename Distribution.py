import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from utils.OrganizeDirectories import organize_directories
import pickle



def is_image(filename):
    try:
        with Image.open(filename):
            return True
    except Exception:
        return False


def count_images(directory):
    counter = Counter()
    lst = os.walk(directory)
    for root, dirs, files in lst:
        category = os.path.basename(root)
        for file in files:
            if is_image(os.path.join(root, file)):
                counter[category] += 1

    print(directory)
    pickle_file = f"utils/{directory}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(counter, f)

    print(counter)

    return counter


def create_charts(data, directory_name):
    if isinstance(data, Counter):
        data = dict(data)

    labels = list(data.keys())
    values = list(data.values())

    # Create 'plots' directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Pie chart
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'{directory_name} class distribution')
    ax1.axis('equal')

    # Bar chart
    ax2.bar(labels, values)
    ax2.set_title(f'{directory_name} class bar chart')
    ax2.set_xlabel('Categories')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{directory_name}_class_charts.png'))
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)

    path = input('Write the directory name where the images are stored.\n'
                 'Example: leaves/images: \n')

    organize_directories("images")

    directory = sys.argv[1]
    directory_name = os.path.basename(directory)

    data = count_images(directory)
    create_charts(data, directory_name)

    print(f"Charts have been saved as {directory_name}"
          " in plot directory")


if __name__ == "__main__":
    main()
