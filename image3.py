import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from collections import Counter
from kmeans import KMeans

np.random.seed(42)

# Clase para procesar imágenes
class ImageProcessor:
    def __init__(self, base_folder, test_folder):
        self.base_folder = base_folder  # Carpeta con las imágenes de entrenamiento
        self.test_folder = test_folder  # Carpeta con las imágenes de prueba

    def read_image(self, file_path):
        if file_path.lower().endswith('.png'):
            return cv2.imread(file_path)
        else:
            print("El archivo no es un .jpg válido.")
            return None

    def remove_background(self, image):
        height, width = image.shape[:2]
        resized_image = cv2.resize(image, (width // 5, height // 5))
        blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv_image, lower_white, upper_white)
        mask_inv = cv2.bitwise_not(mask)
        filtered_mask = np.zeros_like(mask_inv)
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        no_background = cv2.bitwise_and(resized_image, resized_image, mask=filtered_mask)
        return no_background, filtered_mask

    def extract_contours(self, filtered_mask):
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return max(contours, key=cv2.contourArea)
        return None

    def extract_features(self, image, filtered_mask):
        contour = self.extract_contours(filtered_mask)
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_channel = hsv_image[:, :, 0]
            hist = cv2.calcHist([h_channel], [0], filtered_mask, [5], [0, 180]).flatten()
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_2 = hu_moments[1]
            hu_4 = hu_moments[3]
            mean_color = cv2.mean(hsv_image, mask=filtered_mask)[:3]
            return [aspect_ratio, circularity, hu_2, hu_4] + hist.tolist() + list(mean_color)
        return [0] * 12

    def select_k_best(self, features, labels, k=4):
        selector = SelectKBest(score_func=f_classif, k=k)
        self.selector = selector.fit(features, labels)
        reduced_features = self.selector.transform(features)
        return reduced_features

    def transform_features(self, features):
        return self.selector.transform(features)

    def train_kmeans(self):
        feature_vectors = []
        labels = []

        for folder_name in os.listdir(self.base_folder):
            folder_path = os.path.join(self.base_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if not file_name.lower().endswith(('png', 'jpg', 'jpeg', 'dng')):
                    continue

                image = self.read_image(file_path)
                no_background, filtered_mask = self.remove_background(image)
                features = self.extract_features(no_background, filtered_mask)
                feature_vectors.append(features)
                labels.append(folder_name)

        feature_vectors = np.array(feature_vectors)
        labels = np.array(labels)

        # Escalar las características
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_vectors)

        # Selección de características
        reduced_features = self.select_k_best(scaled_features, labels, k=4)

        # Entrenamos el modelo KMeans
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(reduced_features)  # Aseguramos que el modelo se entrene correctamente

        # Mapeo entre el número de cluster y el nombre de la verdura
        cluster_names = {}
        for cluster in range(kmeans.n_clusters):
            members = [labels[i] for i in range(len(labels)) if kmeans.predict([reduced_features[i]])[0] == cluster]
            cluster_names[cluster] = Counter(members).most_common(1)[0][0]  # Asigna el nombre de la verdura al cluster

        return kmeans, scaler, labels, reduced_features, cluster_names  # Devolver también cluster_names

    def classify_test_images(self, kmeans, scaler, labels, reduced_features, cluster_names):
        test_images = []

        # Clasificamos las imágenes de prueba
        for file_name in os.listdir(self.test_folder):
            file_path = os.path.join(self.test_folder, file_name)
            if not file_name.lower().endswith(('png', 'jpg', 'jpeg', 'dng', 'heic')):
                continue

            image = self.read_image(file_path)
            no_background, filtered_mask = self.remove_background(image)
            features = self.extract_features(no_background, filtered_mask)
            scaled_features = scaler.transform([features])
            reduced_features_test = self.transform_features(scaled_features)
            test_cluster = kmeans.predict(reduced_features_test)

            test_images.append((file_name, test_cluster[0]))  # Guardamos el nombre y la predicción

        # Mostrar las imágenes con la predicción del cluster como nombre de verdura
        for image_name, predicted_cluster in test_images:
            image_path = os.path.join(self.test_folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Mostrar la imagen con la predicción del nombre de la verdura
            predicted_label = cluster_names.get(predicted_cluster, 'Desconocido')
            plt.imshow(image)
            plt.title(f"Predicción: {predicted_label}")
            plt.axis('off')
            plt.show()
    
        return test_images  # Devolver la lista de imágenes de prueba

    def visualize_segmentation(self, image_path):
        """Visualiza el procesamiento de segmentación de una imagen"""
        image = self.read_image(image_path)
        if image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            return
        
        no_background, filtered_mask = self.remove_background(image)
        contour = self.extract_contours(filtered_mask)
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Procesamiento de Segmentación: {os.path.basename(image_path)}", fontsize=16)
        
        # 1. Imagen original
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title("Imagen Original")
        axes[0, 0].axis('off')
        
        # 2. Imagen redimensionada
        height, width = image.shape[:2]
        resized_image = cv2.resize(image, (width // 5, height // 5))
        resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(resized_rgb)
        axes[0, 1].set_title("Imagen Redimensionada")
        axes[0, 1].axis('off')
        
        # 3. Máscara de fondo blanco
        blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv_image, lower_white, upper_white)
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title("Máscara Fondo Blanco")
        axes[0, 2].axis('off')
        
        # 4. Máscara invertida
        mask_inv = cv2.bitwise_not(mask)
        axes[1, 0].imshow(mask_inv, cmap='gray')
        axes[1, 0].set_title("Máscara Invertida")
        axes[1, 0].axis('off')
        
        # 5. Máscara filtrada
        axes[1, 1].imshow(filtered_mask, cmap='gray')
        axes[1, 1].set_title("Máscara Filtrada (sin ruido)")
        axes[1, 1].axis('off')
        
        # 6. Imagen sin fondo con contorno dibujado
        no_bg_rgb = cv2.cvtColor(no_background, cv2.COLOR_BGR2RGB)
        result_image = no_bg_rgb.copy()
        if contour is not None:
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
        axes[1, 2].imshow(result_image)
        axes[1, 2].set_title("Imagen Segmentada con Contorno")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Rutas de las carpetas
    base_folder = "/home/bruno/fing/IA_I/data"  # Carpeta con imágenes de entrenamiento
    # test_folder = "/home/bruno/fing/IA_I/img/banana/"  # Carpeta con imágenes de prueba
    test_folder = "/home/bruno/fing/IA_I/data/naranja_sin_fondo/"  # Carpeta con imágenes de prueba
    
    # Crear procesador
    processor = ImageProcessor(base_folder, test_folder)
    
    # Visualizar segmentación de una imagen de prueba
    print("Visualizando segmentación de imágenes de prueba...")
    for file_name in os.listdir(test_folder):
        if file_name.lower().endswith(('jpg', 'jpeg', 'png', 'dng')):
            image_path = os.path.join(test_folder, file_name)
            processor.visualize_segmentation(image_path)
            break  # Mostrar solo la primera imagen


