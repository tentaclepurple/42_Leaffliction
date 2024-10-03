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

        lower_green = np.array([35, 90, 90])
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
    
        
        #last working roi
    """ def roi_objects(self):
        # Asegúrate de que la imagen haya sido cargada correctamente
        if self.image is None:
            raise ValueError("No se ha cargado la imagen correctamente.")

        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Definir rango de color verde en HSV para la hoja
        lower_green = np.array([35, 40, 40])  # Ajusta si es necesario
        upper_green = np.array([85, 255, 255])

        # Crear una máscara que capture el rango de color verde
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Aplicar desenfoque gaussiano para suavizar la imagen y reducir el ruido
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Encontrar los contornos de los objetos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No se encontraron contornos.")
            return self.image.copy()

        # Crear una imagen para visualizar el resultado con el fondo original
        result = self.image.copy()

        # Dibujar todos los contornos encontrados
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        # Dibujar un rectángulo alrededor de toda la región de interés
        x, y, w, h = cv2.boundingRect(np.vstack(contours))  # Unión de todos los contornos
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Retorna la imagen con los contornos y el ROI dibujado
        return result """
 
    
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
            leaf_contour = max(contours, key=cv2.contourArea)
        else:
            return self.image
        
        result = self.image.copy()
        cv2.drawContours(result, [leaf_contour], -1, (255, 0, 255), 2)
        
        bottom_point = tuple(leaf_contour[leaf_contour[:, :, 1].argmax()][0])
        moments = cv2.moments(leaf_contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        angle = np.arctan2(cy - bottom_point[1], cx - bottom_point[0])
        length = 360
        
        start_x = int(bottom_point[0] - length * np.cos(angle))
        start_y = int(bottom_point[1] - length * np.sin(angle))
        end_x = int(bottom_point[0] + length * np.cos(angle))
        end_y = int(bottom_point[1] + length * np.sin(angle))
        
        cv2.line(result, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)
        
        height, width = result.shape[:2]
        center_x = width // 2
        cv2.line(result, (center_x, 0), (center_x, height), (255, 0, 255), 2)
        
        cv2.line(result, (0, 0), (width, 0), (255, 0, 255), 2)
        
        m = (end_y - start_y) / (end_x - start_x) if (end_x - start_x) != 0 else float('inf')
        b = start_y - m * start_x if m != float('inf') else start_x
        intersection_x = center_x
        intersection_y = int(m * intersection_x + b) if m != float('inf') else start_y
        
        cv2.circle(result, (intersection_x, intersection_y), 5, (255, 0, 255), -1)
        
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.2, cv2.NORM_MINMAX)
        _, dist_thresh = cv2.threshold(dist, 0.7, 1.0, cv2.THRESH_BINARY)
        dist_thresh = (dist_thresh * 255).astype(np.uint8)
        contours, _ = cv2.findContours(dist_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

        return result

    """ def pseudolandmarks(self):
        corners = cv2.goodFeaturesToTrack(self.gray, 25, 0.01, 10)
        img_copy = self.image.copy()
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
        return img_copy """



    """ def pseudolandmarks(self):
        img_copy = self.image.copy()
        pseudolandmarks = []

        ### Corner Detection (Red Dots)
        corners = cv2.goodFeaturesToTrack(self.gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
                pseudolandmarks.append(('corner', int(x), int(y)))

        ### Edge Detection (Blue Dots)
        edges = cv2.Canny(self.gray, 100, 200)
        # Find edge points
        edge_points = np.column_stack(np.where(edges > 0))
        # Sample some edge points
        np.random.shuffle(edge_points)
        edge_samples = edge_points[:50]
        for point in edge_samples:
            y, x = point  # Note: y comes first in np.where output
            cv2.circle(img_copy, (x, y), 3, (255, 0, 0), -1)
            pseudolandmarks.append(('edge', x, y))

        ### Contour Sampling (Green Dots)
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            leaf_contour = max(contours, key=cv2.contourArea)
            contour_points = leaf_contour[:, 0, :]
            # Sample points along the contour
            num_points = 50
            indices = np.linspace(0, len(contour_points), num_points, endpoint=False, dtype=int)
            contour_samples = contour_points[indices]
            for point in contour_samples:
                x, y = point
                cv2.circle(img_copy, (int(x), int(y)), 3, (0, 255, 0), -1)
                pseudolandmarks.append(('contour', int(x), int(y)))

        return img_copy """
        
        
    def pseudolandmarks(self):
        img_copy = self.image.copy()
        
        # 1. Detect corners (red)
        corners = cv2.goodFeaturesToTrack(self.gray, 25, 0.01, 10)
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # 2. Detect edges using Canny (green)
        edges = cv2.Canny(self.gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            for point in approx:
                cv2.circle(img_copy, tuple(point[0]), 3, (0, 255, 0), -1)
        
        # 3. Find centroid of the object (blue)
        _, thresh = cv2.threshold(self.gray, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(img_copy, (cX, cY), 3, (255, 0, 0), -1)
        
        return img_copy    
        
    
    def pseudola(self):

        from skimage.filters import frangi

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply Frangi filter to enhance vein-like structures
        veins = frangi(enhanced)
        veins = (veins * 255).astype(np.uint8)

        # Threshold the image to create a binary image
        _, veins_binary = cv2.threshold(veins, 20, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        veins_cleaned = cv2.morphologyEx(veins_binary, cv2.MORPH_OPEN, kernel)
        veins_cleaned = cv2.morphologyEx(veins_cleaned, cv2.MORPH_CLOSE, kernel)

        # Overlay the veins on the original image
        img_veins = self.image.copy()
        img_veins[veins_cleaned > 0] = [0, 0, 255]  # Mark veins in red

        return img_veins



    
    
    


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