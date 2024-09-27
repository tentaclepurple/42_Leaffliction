import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from utils.OrganizeDirectories import organize_directories


def is_image(filename):
    try:
        with Image.open(filename) as img:
            return True
    except:
        return False

def count_images(directory):
    counter = Counter()
    list = os.walk(directory)
    for root, dirs, files in tqdm(list):
        category = os.path.basename(root)
        for file in files:

            if is_image(os.path.join(root, file)):
                counter[category] += 1
    return counter

def create_charts(data, directory_name):
    labels = list(data.keys())
    values = list(data.values())

    # Create 'plots' directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Pie chart
    plt.figure(figsize=(10, 5))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'{directory_name} class distribution')
    plt.axis('equal')
    plt.savefig(f'plots/{directory_name}_pie_chart.png')
    plt.close()

    # Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(f'{directory_name} class bar chart')
    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/{directory_name}_bar_chart.png')
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)

    organize_directories("images")

    directory = sys.argv[1]
    directory_name = os.path.basename(directory)

    data = count_images(directory)
    create_charts(data, directory_name)

    print(f"Charts have been saved as {directory_name}_pie_chart.png and {directory_name}_bar_chart.png in plot directory")

if __name__ == "__main__":
    main()