from plantcv import plantcv as pcv
import cv2
import random
import pickle
import os
import sys
import time


class Augmentation:
    def __init__(self, specie):
        self.specie = specie
        self.img = None

    def get_img(self, path):
        try:
            self.img, mask, metadata = pcv.readimage(path)

        except Exception as e:
            print(f"Error: {str(e)}")

    def save_img(self, save_path):
        try:
            if self.img is not None:
                pcv.print_image(self.img, save_path)
                #print(f"Image saved in {save_path}")
            else:
                print("Couldn't save the image")
        except Exception as e:
            print(f"Error: {str(e)}")

    def rotate(self, percent):
        angle = int((percent / 100) * 360)

        self.img = pcv.transform.rotate(self.img, angle, crop=True)
        return f"Image rotated {angle} degrees"

    def blur(self, percent):
        conv = int((percent / 100) * 30)
        conv = conv if conv % 2 == 1 else conv + 1
        kernel_size = (conv, conv)

        self.img = pcv.gaussian_blur(self.img, kernel_size)
        return f"Applied gaussian blur with kernel size: {kernel_size}"

    def flip(self, percent):
        direction = ""
        if percent % 2 == 0:
            direction = "horizontal"
        else:
            direction = "vertical"
        self.img = pcv.flip(self.img, direction)
        return f"Image flipped. Axis: {direction}"

    def zoom(self, percent):
        zoom_factor = 1 + (percent / 100)
        if self.img is None:
            print("No image loaded")
            return

        # Get the dimensions of the image
        h, w = self.img.shape[:2]

        # Calculate the size of the cropped area
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)

        # Calculate the coordinates to crop around the center
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2

        # Crop the image
        cropped_img = self.img[start_y:start_y + new_h,
                               start_x:start_x + new_w]

        # Resize the cropped image back to the original size
        self.img = cv2.resize(cropped_img, (w, h),
                              interpolation=cv2.INTER_LINEAR)
        return f"Applied a zoom factor of {zoom_factor}"

    def add_contrast(self, percent):
        if self.img is None:
            print("No image loaded")
            return

        # Convert the percentage to an alpha factor
        alpha = 1 + (percent / 100)

        # Apply contrast adjustment (no change in brightness, so beta is 0)
        self.img = cv2.convertScaleAbs(self.img, alpha=alpha, beta=0)
        return f"Added contrast with alpha {alpha}"
    
    def add_brightness(self, percent):
        beta = int((percent / 100) * 255)
        if self.img is None:
            print("No image loaded")
            return

        self.img = cv2.convertScaleAbs(self.img, alpha=1, beta=beta)
        return f"Added {percent}% brightness"

    def chose_rand_method(self):
        aug_methods = [
                    "add_brightness",
                    "add_contrast",
                    "zoom",
                    "flip",
                    "rotate",
                    "blur"
        ]
        aug_method = random.choice(aug_methods)
        perc = random.randint(0, 100)
        msg = getattr(self, aug_method)(perc)
        print(msg)
        return aug_method

    def get_samp_stats(self, dic):
        max_class = max(dic, key=dic.get)
        min_class = min(dic, key=dic.get)

        max_count = dic[max_class]
        min_count = dic[min_class]

        return max_class, max_count, min_class, min_count

    def sampling(self, method):
        file = self.specie + '.pkl'
        file = os.path.join('utils', file)
        
        with open(file, 'rb') as f:
            dic = pickle.load(f)

        max_class, max_count, min_class, min_count = self.get_samp_stats(dic)

        if method == "oversample":
            del dic[max_class]
            for folder, count in dic.items():
                diff = max_count - count
                self.oversample(folder, diff)
        elif method == "undersample":
            del dic[min_class]
            for folder, count in dic.items():
                diff = count - min_count
                self.undersampling(folder, diff)
                
        else:
            print("Sampling method error")

    def oversample(self, folder, diff):
        eval_output = "augmented_directory"
        if not os.path.exists(eval_output):
            os.makedirs(eval_output)
        path = f"{self.specie}/{folder}"
        for _, _, files in os.walk(path):
            for i in range(diff):
                file = random.choice(files)
                name, ext = os.path.splitext(file)
                img_path = f"{path}/{file}"
                self.get_img(img_path)
                aug_meth = self.chose_rand_method()
                eval_path = f"{eval_output}/{name}_{aug_meth}_{i}{ext}"
                same_path = f"{path}/{name}_{aug_meth}_{i}{ext}"
                self.save_img(eval_path)
                self.save_img(same_path)

    def undersampling(self, folder, diff):
        for _, _, files in os.walk(f"{self.specie}/{folder}"):
            for i in range(diff):
                path_to_remove = f"{self.specie}/{folder}/{files[i]}" 
                print(f"Removing {files[i]} from {folder}")
                os.remove(path_to_remove)


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            print("Usage: python Augmentation.py <specie>")
            sys.exit(1)

        aug = None
        specie = sys.argv[1]

        print("Chose a sampling method: 1 for oversample or 2 for undersample")
        method = input()
        
        if method == "1":
            method = "oversample"
        elif method == "2":
            method = "undersample"
        else:
            print("Invalid method")
            sys.exit(1)
        
        aug = Augmentation(specie)
        aug.sampling(method)
    except Exception as e:
        print(f"Error: {str(e)}")
