import cv2
import numpy as np
import os


class FruitImage:
    def __init__(self, path):
        self.path = path
        self.bgr = cv2.imread(path)
        if self.bgr is None:
            raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")
        self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)

    # def get_fruit_mask1(self, sat_threshold=40, central_fraction=0.6):
    #     """
    #     Devuelve una máscara booleana donde True = píxel que consideramos fruta.
    #     Estrategia MUY simple:
    #     - Nos quedamos solo con zona central de la imagen.
    #     - Dentro de esa zona, descartamos píxeles de muy baja saturación (fondo claro/madera).
    #     """
    #     h, w = self.hsv.shape[:2]

    #     # Región central
    #     cy0 = int(h * (1 - central_fraction) / 2)
    #     cy1 = int(h * (1 + central_fraction) / 2)
    #     cx0 = int(w * (1 - central_fraction) / 2)
    #     cx1 = int(w * (1 + central_fraction) / 2)

    #     central_mask = np.zeros((h, w), dtype=bool)
    #     central_mask[cy0:cy1, cx0:cx1] = True

    #     # Filtro por saturación (canal S)
    #     s_channel = self.hsv[:, :, 1]
    #     sat_mask = s_channel > sat_threshold

    #     fruit_mask = central_mask & sat_mask

    #     # Si casi no hay píxeles, usamos solo la región central sin filtro de saturación
    #     if np.count_nonzero(fruit_mask) < 50:
    #         fruit_mask = central_mask

    #     return fruit_mask

    def get_fruit_mask(
        self,
        sat_threshold=40,
        central_fraction=0.6,
        use_bg_hue=True,
        bg_border_frac=0.1,
        bg_hue_tolerance=2,
    ):
        """
        Devuelve una máscara booleana donde True = píxel que consideramos fruta.

        Estrategia:
        - Zona central de la imagen (asumimos fruta centrada).
        - Filtro por saturación (descarta fondo blanco / muy apagado).
        - (Opcional) Estimar Hue del fondo con los bordes y descartar píxeles con Hue parecido.
          Esto ayuda cuando el fondo es madera marrón.
        """
        h, w = self.hsv.shape[:2]

        # Región central
        cy0 = int(h * (1 - central_fraction) / 2)
        cy1 = int(h * (1 + central_fraction) / 2)
        cx0 = int(w * (1 - central_fraction) / 2)
        cx1 = int(w * (1 + central_fraction) / 2)

        central_mask = np.zeros((h, w), dtype=bool)
        central_mask[cy0:cy1, cx0:cx1] = True

        # Filtro por saturación (canal S)
        s_channel = self.hsv[:, :, 1]
        sat_mask = s_channel > sat_threshold

        fruit_mask = central_mask & sat_mask

        # --- estimar fondo por Hue en los bordes ---
        if use_bg_hue:
            h_channel = self.hsv[:, :, 0].astype(np.float32)

            bf = bg_border_frac
            border_mask = np.zeros((h, w), dtype=bool)
            # bandas superior, inferior, izquierda y derecha
            border_mask[:int(h*bf), :] = True
            border_mask[h-int(h*bf):, :] = True
            border_mask[:, :int(w*bf)] = True
            border_mask[:, w-int(w*bf):] = True

            bg_h_vals = h_channel[border_mask]
            if len(bg_h_vals) > 0:
                bg_hue = float(np.median(bg_h_vals))

                # píxeles cuyo Hue se aleja lo suficiente del fondo
                hue_diff = np.abs(h_channel - bg_hue)
                # cuidado con circularidad de Hue (0 y 179 son vecinos)
                hue_diff = np.minimum(hue_diff, 180 - hue_diff)

                not_bg_mask = hue_diff > bg_hue_tolerance

                fruit_mask = fruit_mask & not_bg_mask

        # Si casi no hay píxeles, usamos solo la región central sin filtros
        if np.count_nonzero(fruit_mask) < 50:
            fruit_mask = central_mask

        return fruit_mask


    def get_features(self, sat_threshold=40, central_fraction=0.6):
        """
        Extrae features completos de la fruta:
        - cos(H), sin(H): Hue circular
        - S, V: Saturación y Valor normalizados [0, 1]
        - Hu Moments: 2 momentos de forma normalizados
        - Circularidad: normalizada con log
        - Compacidad: normalizada con log
        
        Devuelve una tupla de 8 valores: (cos_h, sin_h, S, V, hu0, hu1, circ, compact)
        """
        mask = self.get_fruit_mask(
            sat_threshold=sat_threshold,
            central_fraction=central_fraction
        )

        h_channel = self.hsv[:, :, 0].astype(np.float32)
        s_channel = self.hsv[:, :, 1].astype(np.float32)
        v_channel = self.hsv[:, :, 2].astype(np.float32)

        h_vals = h_channel[mask]
        s_vals = s_channel[mask]
        v_vals = v_channel[mask]

        # 1. COLOR: Hue circular + S y V normalizados
        h_rad = h_vals * (2 * np.pi / 180.0)
        cos_h = float(np.mean(np.cos(h_rad)))
        sin_h = float(np.mean(np.sin(h_rad)))
        sat_med = float(np.median(s_vals)) / 255.0
        val_med = float(np.median(v_vals)) / 255.0

        # 2. FORMA: Crear máscara binaria para contorno
        gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Calcular momentos de Hu
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Seleccionar Hu[0] y Hu[1] y normalizar con log
        d0 = 1e-8  # Evitar log(0)
        hu0_norm = float(np.log10(np.abs(hu_moments[0]) + d0))
        hu1_norm = float(np.log10(np.abs(hu_moments[1]) + d0))
        
        # Normalizar Hu moments a [0, 1] aproximadamente
        # Rangos típicos de log10(Hu): [-8, 0]
        hu0_scaled = (hu0_norm + 8.0) / 8.0  # Escalar de [-8,0] a [0,1]
        hu1_scaled = (hu1_norm + 8.0) / 8.0
        hu0_scaled = np.clip(hu0_scaled, 0, 1)
        hu1_scaled = np.clip(hu1_scaled, 0, 1)
        
        # Encontrar contornos para circularidad y compacidad
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularidad: (4π * área) / perímetro²
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            
            # Compacidad: perímetro² / área
            if area > 0:
                compactness = (perimeter ** 2) / area
            else:
                compactness = 0
            
            # Normalizar con log y escalar a [0, 1]
            circ_log = np.log10(np.abs(circularity) + d0)
            compact_log = np.log10(np.abs(compactness) + d0)
            
            # Rangos típicos: circularity [-2, 0], compactness [1, 3]
            circ_scaled = (circ_log + 2.0) / 2.0  # [-2,0] → [0,1]
            compact_scaled = (compact_log - 1.0) / 2.0  # [1,3] → [0,1]
            
            circ_scaled = float(np.clip(circ_scaled, 0, 1))
            compact_scaled = float(np.clip(compact_scaled, 0, 1))
        else:
            circ_scaled = 0.0
            compact_scaled = 0.0

        return cos_h, sin_h, sat_med, val_med, hu0_scaled, hu1_scaled, circ_scaled, compact_scaled
    
    def get_dominant_hsv(self, sat_threshold=40, central_fraction=0.6):
        """
        DEPRECATED: Usa get_features() en su lugar.
        Mantiene compatibilidad con código anterior.
        """
        feats = self.get_features(sat_threshold, central_fraction)
        return feats[:4]  # Devuelve solo cos_h, sin_h, s, v




class FruitDataset:
    """
    Carga un conjunto de imágenes organizadas en carpetas por fruta y
    construye un dataset de vectores [cos(H), sin(H), S, V, hu0, hu1, circ, compact] + etiquetas.
    """

    def __init__(self, root_dir):
        """
        root_dir: carpeta raíz que contiene una subcarpeta por fruta.
        Ejemplo:
            root_dir/
                manzana/
                banana/
                pera/
                naranja/
        """
        self.root_dir = root_dir
        self.X = []  # lista de vectores [cos(H), sin(H), S, V, hu0, hu1, circ, compact]
        self.y = []  # lista de etiquetas (strings, ej: "manzana")

    def build(self, sat_threshold=40, central_fraction=0.6):
        """
        Recorre todas las subcarpetas, procesa las imágenes
        y completa X e y.
        """
        self.X = []
        self.y = []

        # Recorremos cada subcarpeta dentro de root_dir
        for label in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_path):
                continue  # ignorar archivos sueltos

            # label = nombre de la fruta (manzana, banana, etc.)
            print(f"Procesando clase: {label}")

            for filename in os.listdir(label_path):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(label_path, filename)

                try:
                    fruit_img = FruitImage(img_path)
                    features = fruit_img.get_features(
                        sat_threshold=sat_threshold,
                        central_fraction=central_fraction
                    )
                    self.X.append(list(features))
                    self.y.append(label)
                except FileNotFoundError as e:
                    print(e)

        # Convertimos a arrays de numpy por comodidad (opcional)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y)

    def get_data(self):
        """
        Devuelve (X, y):
        - X: array de shape (n_muestras, 8) con features [cos(H), sin(H), S, V, hu0, hu1, circ, compact]
        - y: array de etiquetas de longitud n_muestras
        """
        return self.X, self.y



if __name__ == "__main__":
    root = "/home/bruno/fing/IA_I/data_sin_fondo/"  # ruta a tu carpeta raíz

    ds = FruitDataset(root)
    ds.build(sat_threshold=40, central_fraction=0.6)

    X, y = ds.get_data()

    print("Shape X:", X.shape)  # (n_imágenes, 4)
    print("Etiquetas:", y)

    # Ejemplo: mostrar todas las muestras
    for i in range(len(X)):
        cos_h, sin_h, s, v, hu0, hu1, circ, compact = X[i]
        # Opcional: reconstruir el ángulo Hue para visualizar
        hue_reconstructed = np.arctan2(sin_h, cos_h) * (180.0 / np.pi)
        if hue_reconstructed < 0:
            hue_reconstructed += 360
        print(f"{i}: label={y[i]}, H≈{hue_reconstructed:.0f}°, S={s:.2f}, V={v:.2f}, Hu0={hu0:.2f}, Hu1={hu1:.2f}, Circ={circ:.2f}, Comp={compact:.2f}")
