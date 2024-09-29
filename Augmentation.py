from plantcv import plantcv as pcv
import cv2
from utils.Sampling import *

class Augmentation:
    def __init__(self, path):
        self.path = path
        self.img = None
        try:
            print(path)
            self.img, mask, metadata = pcv.readimage(self.path)
            print(f"Metadata: {metadata}")

        except Exception as e:
            print(f"Error: {str(e)}")

    def save(self, save_path):
        try:
            if self.img is not None:
                pcv.print_image(self.img, save_path)
                print(f"Image saved in {save_path}")
            else:
                print("Couldn't save the image")
        except Exception as e:
            print(f"Error: {str(e)}")

    def rotate(self, percent):
        angle = int((percent / 100) * 360)
        print(angle)

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


if __name__ == '__main__':
    
    """ aug = None
    file = "image1.JPG"
    save_folder = "tests"
    
    aug = Augmentation(f"{file}")
    print(aug.blur(150))
    aug.save(f"{save_folder}/{file}")
 """
    
    #sampling("Apple", "oversample")
    sampling("Grape", "undersample")