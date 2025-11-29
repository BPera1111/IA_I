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

    def get_fruit_mask(self, sat_threshold=40, central_fraction=0.6):
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

    def get_dominant_hsv(self, sat_threshold=40, central_fraction=0.6):
        """
        Calcula H, S, V medianos de los píxeles que consideramos fruta.
        Devuelve una tupla (H, S, V) lista para usar como vector de características.
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

        hue_med = float(np.median(h_vals))
        sat_med = float(np.median(s_vals))
        val_med = float(np.median(v_vals))

        return hue_med, sat_med, val_med




class FruitDataset:
    """
    Carga un conjunto de imágenes organizadas en carpetas por fruta y
    construye un dataset de vectores [H, S, V] + etiquetas.
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
        self.X = []  # lista de vectores [H, S, V]
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
                    h, s, v = fruit_img.get_dominant_hsv(
                        sat_threshold=sat_threshold,
                        central_fraction=central_fraction
                    )
                    self.X.append([h, s, v])
                    self.y.append(label)
                except FileNotFoundError as e:
                    print(e)

        # Convertimos a arrays de numpy por comodidad (opcional)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y)

    def get_data(self):
        """
        Devuelve (X, y):
        - X: array de shape (n_muestras, 3)
        - y: array de etiquetas de longitud n_muestras
        """
        return self.X, self.y



if __name__ == "__main__":
    root = "/home/bruno/fing/IA_I/img/"  # ruta a tu carpeta raíz

    ds = FruitDataset(root)
    ds.build(sat_threshold=40, central_fraction=0.6)

    X, y = ds.get_data()

    print("Shape X:", X.shape)  # (n_imágenes, 3)
    print("Etiquetas:", y)

    # Ejemplo: mostrar primeros 5
    for i in range(min(8, len(X))):
        print(f"{i}: vec = {X[i]}, label = {y[i]}")
