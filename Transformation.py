import cv2
from plantcv import plantcv as pcv
import numpy as np
import os
import argparse
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



class Transformation:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.mask = None
        
    def gaussian_blur(self):
        if self.image is None:
            self.load_original()

        s = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")

        s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=110, object_type="light"
        )
        gauss = pcv.gaussian_blur(
            img=s_thresh, ksize=(3, 3), sigma_x=0, sigma_y=None
        )
        
        return gauss
    
    #GOOD for now
    """ def create_mask(self):
        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Definir rango de color verde en HSV
        lower_green = np.array([50, 40, 40])
        upper_green = np.array([90, 255, 255])

        # Crear una máscara para las partes verdes
        self.mask = cv2.inRange(hsv, lower_green, upper_green)

        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((3, 3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)

        # Invertir la máscara
        mask_inv = cv2.bitwise_not(self.mask)
        
        # Crear una imagen en blanco del mismo tamaño que la original
        white_bg = np.full(self.image.shape, 255, dtype=np.uint8)

        # Usar la máscara para combinar la imagen original con el fondo blanco
        result = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
        result = cv2.add(result, cv2.bitwise_and(white_bg, white_bg, mask=self.mask))

        return result """
        
        
    def create_mask(self):
        # Gaussian mask for bg
        s = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")
        s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=50, object_type="light"
            )
        gaussmask = pcv.gaussian_blur(
            img=s_thresh, ksize=(3, 3), sigma_x=0, sigma_y=None
        )

        # Mask for leaf
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([120, 255, 255])
        self.mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((1, 1), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        mask_inv = cv2.bitwise_not(self.mask)


        combined_mask = cv2.bitwise_and(mask_inv, gaussmask)

        masked_image = pcv.apply_mask(
            img=self.image,
            mask=combined_mask,
            mask_color="white"
            )

        return masked_image

    def roi_objects(self):
        if self.image is None:
            raise ValueError("No se ha cargado la imagen correctamente.")

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 85, 85])
        upper_green = np.array([85, 255, 255])

        lower_brown = np.array([10, 40, 40])
        upper_brown = np.array([40, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

        leaf_mask = cv2.bitwise_or(green_mask, brown_mask)

        kernel = np.ones((13, 13), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

        non_green_mask = cv2.bitwise_not(green_mask)

        non_green_in_leaf = cv2.bitwise_and(non_green_mask, non_green_mask, mask=leaf_mask)

        result = self.image.copy()
        result[non_green_in_leaf > 0] = [0, 255, 0]

        contours, _ = cv2.findContours(
            leaf_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
            )
        if not contours:
            print("No contours found.")
            return self.image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        # draw a rectangle around the region of interest
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)

        return result


    def analyze_object(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            return cv2.drawContours(self.image.copy(), [main_contour], 0, (255, 0, 0), 2)
        return self.image.copy()

    def pseudolandmarks(self):
        corners = cv2.goodFeaturesToTrack(self.gray, 25, 0.01, 10)
        img_copy = self.image.copy()
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
        return img_copy

    def color_histogram(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

        channels = [
            (self.image[:,:,0], 'blue'),
            ((self.image[:,:,0] + self.image[:,:,2])/2, 'blue-yellow'),
            (self.image[:,:,1], 'green'),
            ((self.image[:,:,1] + self.image[:,:,2])/2, 'green-magenta'),
            (self.image[:,:,2], 'red'),
            (hsv[:,:,0], 'hue'),
            (lab[:,:,0], 'lightness'),
            (hsv[:,:,1], 'saturation'),
            (hsv[:,:,2], 'value')
        ]

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['blue', 'yellow', 'green', 'magenta', 'red', 'purple', 'gray', 'cyan', 'orange']

        for (channel, name), color in zip(channels, colors):
            hist, _ = np.histogram(channel, bins=256, range=[0, 256])
            hist = hist / hist.sum() * 100 
            ax.plot(hist, color=color, label=name, alpha=0.7)

        ax.set_xlim([0, 255])
        ax.set_xlabel('Pixel intensity')
        ax.set_ylabel('Proportion of pixels (%)')
        ax.set_title('Color Histogram')
        ax.grid(True, alpha=0.3)

        ax.legend(title='Color Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig

    def apply_transformations(self, transformations):
        results = {'Original': self.image}
        
        if 'blur' in transformations:
            results['GaussianBlur'] = self.gaussian_blur()
        if 'mask' in transformations:
            results['Mask'] = self.create_mask()
        if 'roi' in transformations:
            results['ROIobjects'] = self.roi_objects()
        if 'analyze' in transformations:
            results['AnalyzeObject'] = self.analyze_object()
        if 'landmarks' in transformations:
            results['Pseudolandmarks'] = self.pseudolandmarks()
        
        if 'histogram' in transformations:
            results['ColorHistogram'] = self.color_histogram()
        
        return results

def process_and_save(image_path, transformations, save_dir=None):
    trans = Transformation(image_path)
    results = trans.apply_transformations(transformations)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, img in results.items():
            if name == 'ColorHistogram':
                img.savefig(os.path.join(save_dir, f"{name}.png"))
                plt.close(img)
            else:
                cv2.imwrite(os.path.join(save_dir, f"{name}.png"), img)
    else:
        plt.figure(figsize=(20, 10))
        for i, (name, img) in enumerate(results.items(), 1):
            if name != 'ColorHistogram':
                plt.subplot(2, 3, i)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(name)
                plt.axis('off')
            else:
                plt.subplot(2, 3, i)
                plt.imshow(img)
                plt.title(name)
                plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Apply image transformations for leaf analysis.')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('-dst', '--destination', help='Destination directory for output')
    parser.add_argument('-t', '--transformations', nargs='+', 
                        choices=['blur', 'mask', 'roi', 'analyze', 'landmarks', 'histogram'],
                        default=['blur', 'mask', 'roi', 'analyze', 'landmarks', 'histogram'],
                        help='Transformations to apply')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        process_and_save(args.input, args.transformations, args.destination)
    elif os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                image_path = os.path.join(args.input, filename)
                if args.destination:
                    save_dir = os.path.join(args.destination, os.path.splitext(filename)[0])
                else:
                    save_dir = None
                process_and_save(image_path, args.transformations, save_dir)
    else:
        print("Invalid input. Please provide a valid image file or directory.")

if __name__ == "__main__":
    main()