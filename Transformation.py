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
            gray_img=s, threshold=60, object_type="light"
        )
        gauss = pcv.gaussian_blur(
            img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
        )
        
        return gauss
    
    """ def create_mask(self):
        _, self.mask = cv2.threshold(self.gray, 120, 255, cv2.THRESH_BINARY)
        return self.mask """
        
    def create_mask(self):
        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Definir rango de color verde en HSV
        lower_green = np.array([50, 40, 40])
        upper_green = np.array([90, 255, 255])

        # Crear una máscara para las partes verdes
        self.mask = cv2.inRange(hsv, lower_green, upper_green)

        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((3,3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)

        # Invertir la máscara
        mask_inv = cv2.bitwise_not(self.mask)

        # Crear una imagen en blanco del mismo tamaño que la original
        white_bg = np.full(self.image.shape, 255, dtype=np.uint8)

        # Usar la máscara para combinar la imagen original con el fondo blanco
        result = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
        result = cv2.add(result, cv2.bitwise_and(white_bg, white_bg, mask=self.mask))

        return result

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
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

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
    
    def roi_objects(self):
        # Asegúrate de que la imagen haya sido cargada correctamente
        if self.image is None:
            raise ValueError("No se ha cargado la imagen correctamente.")

        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Definir rango de color verde en HSV para la hoja
        lower_green = np.array([35, 40, 40])  # Ajusta si es necesario
        upper_green = np.array([85, 255, 255])

        # Crear una máscara que capture el rango de color verde
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Invertir la máscara para obtener las áreas no verdes (enfermas)
        mask_diseased = cv2.bitwise_not(mask_green)

        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5, 5), np.uint8)
        mask_diseased = cv2.morphologyEx(mask_diseased, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_diseased = cv2.morphologyEx(mask_diseased, cv2.MORPH_OPEN, kernel, iterations=2)

        # Aplicar desenfoque gaussiano para suavizar la imagen y reducir el ruido
        mask_diseased = cv2.GaussianBlur(mask_diseased, (5, 5), 0)

        # Encontrar los contornos de las áreas enfermas en la máscara
        contours_diseased, _ = cv2.findContours(mask_diseased, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Si no se encontraron contornos de áreas enfermas, avisar
        if not contours_diseased:
            print("No se encontraron áreas enfermas.")
            return self.image.copy()

        # Crear una imagen para visualizar el resultado con el fondo original
        result = self.image.copy()

        # Dibujar todos los contornos de las áreas enfermas en verde
        cv2.drawContours(result, contours_diseased, -1, (0, 255, 0), 2)

        # Encontrar los contornos de la hoja usando la máscara verde
        contours_leaf, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours_leaf:
            print("No se encontraron contornos de la hoja.")
            return result

        # Dibujar un rectángulo alrededor de la hoja
        x, y, w, h = cv2.boundingRect(np.vstack(contours_leaf))  # Unión de todos los contornos de la hoja
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Retorna la imagen con los contornos de las áreas enfermas y el ROI de la hoja dibujados
        return result


    """ def analyze_object(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            leaf_contour = max(contours, key=cv2.contourArea)
        else:
            return self.image
        result = self.image.copy()
        cv2.drawContours(result, [leaf_contour], -1, (255, 0, 255), 2)
        moments = cv2.moments(leaf_contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
        length = max(result.shape) // 2
        end_point1 = (int(cx - length * np.cos(angle)), int(cy - length * np.sin(angle)))
        end_point2 = (int(cx + length * np.cos(angle)), int(cy + length * np.sin(angle)))
        cv2.line(result, end_point1, end_point2, (255, 0, 255), 2)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [leaf_contour], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.2, cv2.NORM_MINMAX)
        _, dist_thresh = cv2.threshold(dist, 0.7, 1.0, cv2.THRESH_BINARY)
        dist_thresh = (dist_thresh * 255).astype(np.uint8)
        contours, _ = cv2.findContours(dist_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

        return result """
    
    """ def analyze_object(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            leaf_contour = max(contours, key=cv2.contourArea)
        else:
            return self.image
        result = self.image.copy()
        cv2.drawContours(result, [leaf_contour], -1, (255, 0, 255), 2)
        
        # Encontrar el punto más bajo del contorno (probablemente el rabo)
        bottom_point = tuple(leaf_contour[leaf_contour[:, :, 1].argmax()][0])
        
        # Calcular los momentos y el centro de masa
        moments = cv2.moments(leaf_contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Calcular el ángulo y la longitud basados en el rabo
        angle = np.arctan2(cy - bottom_point[1], cx - bottom_point[0])
        length = 360
        #length = max(result.shape)  # Extender la línea a través de toda la imagen
        print(length)
        
        # Calcular los puntos finales de la línea
        start_x = int(bottom_point[0] - length * np.cos(angle))
        start_y = int(bottom_point[1] - length * np.sin(angle))
        end_x = int(bottom_point[0] + length * np.cos(angle))
        end_y = int(bottom_point[1] + length * np.sin(angle))
        
        # Dibujar la línea
        cv2.line(result, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)
        
        # Código para las líneas azules interiores (sin cambios)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [leaf_contour], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.2, cv2.NORM_MINMAX)
        _, dist_thresh = cv2.threshold(dist, 0.7, 1.0, cv2.THRESH_BINARY)
        dist_thresh = (dist_thresh * 255).astype(np.uint8)
        contours, _ = cv2.findContours(dist_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

        return result """

    """ def analyze_object(self):
        #working most of times with
        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Definir rango de color verde en HSV
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Crear máscara para el color verde
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            leaf_contour = max(contours, key=cv2.contourArea)
        else:
            return self.image
        
        result = self.image.copy()
        cv2.drawContours(result, [leaf_contour], -1, (255, 0, 255), 2)
        
        # Encontrar el punto más bajo del contorno (probablemente el rabo)
        bottom_point = tuple(leaf_contour[leaf_contour[:, :, 1].argmax()][0])
        
        # Calcular los momentos y el centro de masa
        moments = cv2.moments(leaf_contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Calcular el ángulo y la longitud basados en el rabo
        angle = np.arctan2(cy - bottom_point[1], cx - bottom_point[0])
        length = 360
        
        # Calcular los puntos finales de la línea
        start_x = int(bottom_point[0] - length * np.cos(angle))
        start_y = int(bottom_point[1] - length * np.sin(angle))
        end_x = int(bottom_point[0] + length * np.cos(angle))
        end_y = int(bottom_point[1] + length * np.sin(angle))
        
        # Dibujar la línea
        cv2.line(result, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)
        
        # Código para las líneas azules interiores
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.2, cv2.NORM_MINMAX)
        _, dist_thresh = cv2.threshold(dist, 0.7, 1.0, cv2.THRESH_BINARY)
        dist_thresh = (dist_thresh * 255).astype(np.uint8)
        contours, _ = cv2.findContours(dist_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

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
        
        # Línea diagonal (eje principal)
        cv2.line(result, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)
        
        # Línea vertical en el centro de la imagen
        height, width = result.shape[:2]
        center_x = width // 2
        cv2.line(result, (center_x, 0), (center_x, height), (255, 0, 255), 2)
        
        # Línea horizontal (parte superior de la T)
        cv2.line(result, (0, 0), (width, 0), (255, 0, 255), 2)
        
        # Encontrar punto de intersección
        m = (end_y - start_y) / (end_x - start_x) if (end_x - start_x) != 0 else float('inf')
        b = start_y - m * start_x if m != float('inf') else start_x
        intersection_x = center_x
        intersection_y = int(m * intersection_x + b) if m != float('inf') else start_y
        
        # Dibujar círculo en la intersección
        cv2.circle(result, (intersection_x, intersection_y), 5, (255, 0, 255), -1)
        
        # Código para las líneas azules interiores (sin cambios)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.2, cv2.NORM_MINMAX)
        _, dist_thresh = cv2.threshold(dist, 0.7, 1.0, cv2.THRESH_BINARY)
        dist_thresh = (dist_thresh * 255).astype(np.uint8)
        contours, _ = cv2.findContours(dist_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

        return result


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