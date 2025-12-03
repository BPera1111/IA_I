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

    def get_fruit_mask1(self, sat_threshold=40, central_fraction=0.6):
        """
        Devuelve una máscara booleana donde True = píxel que consideramos fruta.
        Estrategia MUY simple:
        - Nos quedamos solo con zona central de la imagen.
        - Dentro de esa zona, descartamos píxeles de muy baja saturación (fondo claro/madera).
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

        # Si casi no hay píxeles, usamos solo la región central sin filtro de saturación
        if np.count_nonzero(fruit_mask) < 50:
            fruit_mask = central_mask

        return fruit_mask

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


    def get_dominant_hsv(self, sat_threshold=40, central_fraction=0.6):
        """
        Calcula características de color de los píxeles que consideramos fruta.
        
        Para el Hue (circular), usamos representación en coordenadas circulares:
        - cos(H) y sin(H) para manejar la circularidad (rojo está en 0° y 180°)
        
        Devuelve una tupla (cos_h, sin_h, S, V) lista para usar como vector de características.
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

        # Convertir Hue a radianes (OpenCV usa 0-179 para Hue)
        # 179 en OpenCV = 360° = 2π radianes
        h_rad = h_vals * (2 * np.pi / 180.0)
        
        # Calcular coordenadas circulares (promedio de cos y sin)
        cos_h = float(np.mean(np.cos(h_rad)))
        sin_h = float(np.mean(np.sin(h_rad)))
        
        # Para S y V usamos mediana y NORMALIZAMOS a [0, 1]
        # OpenCV usa S y V en rango [0, 255]
        sat_med = float(np.median(s_vals)) / 255.0
        val_med = float(np.median(v_vals)) / 255.0

        return cos_h, sin_h, sat_med, val_med




class FruitDataset:
    """
    Carga un conjunto de imágenes organizadas en carpetas por fruta y
    construye un dataset de vectores [cos(H), sin(H), S, V] + etiquetas.
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
        self.X = []  # lista de vectores [cos(H), sin(H), S, V]
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
                    cos_h, sin_h, s, v = fruit_img.get_dominant_hsv(
                        sat_threshold=sat_threshold,
                        central_fraction=central_fraction
                    )
                    self.X.append([cos_h, sin_h, s, v])
                    self.y.append(label)
                except FileNotFoundError as e:
                    print(e)

        # Convertimos a arrays de numpy por comodidad (opcional)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y)

    def get_data(self):
        """
        Devuelve (X, y):
        - X: array de shape (n_muestras, 4) con features [cos(H), sin(H), S, V]
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
        cos_h, sin_h, s, v = X[i]
        # Opcional: reconstruir el ángulo Hue para visualizar
        hue_reconstructed = np.arctan2(sin_h, cos_h) * (180.0 / np.pi)
        if hue_reconstructed < 0:
            hue_reconstructed += 360
        print(f"{i}: label={y[i]}, cos(H)={cos_h:.3f}, sin(H)={sin_h:.3f}, S={s:.1f}, V={v:.1f} [Hue≈{hue_reconstructed:.1f}°]")
