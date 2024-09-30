import cv2
from plantcv import plantcv as pcv
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

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
            gray_img=s, threshold=60, object_type="light"
        )
        gauss = pcv.gaussian_blur(
            img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
        )
        
        return gauss
    
    """ def create_mask(self):
        _, self.mask = cv2.threshold(self.gray, 120, 255, cv2.THRESH_BINARY)
        return self.mask """
        
    """ def create_mask(self):
        # Aplicar desenfoque Gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        
        # Aplicar umbral de Otsu
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para limpiar la máscara
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Dilatación para asegurar que capturamos toda la hoja
        dilated = cv2.dilate(opening, kernel, iterations=3)
        
        # Invertir la máscara para que la hoja sea blanca y el fondo negro
        mask_inv = cv2.bitwise_not(dilated)
        
        # Crear una imagen en blanco del mismo tamaño que la original
        white_bg = np.full(self.image.shape, 255, dtype=np.uint8)
        
        # Usar la máscara para combinar la imagen original con el fondo blanco
        result = cv2.bitwise_and(self.image, self.image, mask=dilated)
        result = cv2.add(result, cv2.bitwise_and(white_bg, white_bg, mask=mask_inv))
        
        return result """
        
    def create_mask(self):
        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Definir rango de color verde en HSV
        lower_green = np.array([50, 40, 40])
        upper_green = np.array([90, 255, 255])

        # Crear una máscara para las partes verdes
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Invertir la máscara
        mask_inv = cv2.bitwise_not(mask)

        # Crear una imagen en blanco del mismo tamaño que la original
        white_bg = np.full(self.image.shape, 255, dtype=np.uint8)

        # Usar la máscara para combinar la imagen original con el fondo blanco
        result = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
        result = cv2.add(result, cv2.bitwise_and(white_bg, white_bg, mask=mask))

        return result

    def roi_objects(self):
        if self.mask is None:
            self.create_mask()
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.drawContours(self.image.copy(), contours, -1, (0, 255, 0), 2)

    def analyze_object(self):
        if self.mask is None:
            self.create_mask()
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        colors = ('b', 'g', 'r')
        plt.figure()
        for i, color in enumerate(colors):
            hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        return plt

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
            plt.figure()
            self.color_histogram()
            results['ColorHistogram'] = plt
        
        return results

def process_and_save(image_path, transformations, save_dir=None):
    trans = Transformation(image_path)
    results = trans.apply_transformations(transformations)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, img in results.items():
            if name == 'Color histogram':
                img.savefig(os.path.join(save_dir, f"{name}.JPG"))
                plt.close()
            else:
                cv2.imwrite(os.path.join(save_dir, f"{name}.JPG"), img)
    else:
        plt.figure(figsize=(20, 10))
        for i, (name, img) in enumerate(results.items(), 1):
            if name != 'Color histogram':
                plt.subplot(2, 3, i)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(name)
                plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        if 'histogram' in transformations:
            results['Color histogram'].show()

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