import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.blur = None
        self.sharpened = None
        self.edges = None
        self.contorned = None
        self.filled_img = None
        pass

    def get_features(self):

        # Llevamos la imagen de 3 canales a 1 canal
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Hacemos una mascara para eliminar el fondo
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Invertir la máscara para que el fondo sea negro y las frutas sean blancas
        mask = cv2.bitwise_not(mask)

        # Aplicar la máscara a la imagen original para eliminar el fondo
        result = cv2.bitwise_and(self.image, self.image, mask=mask)

        # Redimensiono la imagen
        img = cv2.resize(result, (200, 200))

        # Imagen en rgb
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Hacemos un desenfoque 
        blurred = cv2.bilateralFilter(img, 7, 25, 25)
        self.blur = blurred
        
        # Afila la imagen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        self.sharpened = sharpened
        
        # Calculo de limites para Canny
        median = np.median(sharpened)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))

        # Deteccion de bordes
        pre_edges = cv2.Canny(img, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(pre_edges, cv2.MORPH_CLOSE, kernel)
        self.edges = edges

        # Buscamos contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorned = np.ones((200, 200), np.uint8)
        cv2.drawContours(contorned, contours, -1, (255, 0, 0), 3)

        # # Afila la imagen
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # sharpened = cv2.filter2D(contorned, -1, kernel)

        # Calculamos el area y circularidad
        cont = contours[0]
        max_area = 0
        for cont in contours:                                                                       
            area = cv2.contourArea(cont)
            if area > max_area:                                                                    
                max_area = area
            biggest_cont = cont

        # Calcular el tamaño de la ventana para el filtro gaussiano
        window_size = 3
        gaussian_window = cv2.getGaussianKernel(ksize=window_size, sigma=-1)
        
        # Suavizar el contorno con un filtro gaussiano
        smooth = cv2.sepFilter2D(src=np.float32(biggest_cont), ddepth=-1, kernelX=gaussian_window, kernelY=gaussian_window)
        smooth = np.int32(smooth)

        # Crear una imagen en blanco del mismo tamaño que la imagen original
        smooth_img = np.ones((200, 200), np.uint8)

        # Dibujar el contorno suavizado en la imagen en blanco
        cv2.drawContours(smooth_img, [smooth], -1, (255,0,0), 3)
        self.contorned = smooth_img

        # Calcular el perímetro del contorno
        if biggest_cont is not None:    
            per = cv2.arcLength(smooth, True)                                        
            if per == 0:
                circ = 0
            else:
                circ = 4*np.pi*(max_area/(per*per))     

        # Calcular la compacidad del contorno
        if max_area!=0:
            compactness = per*per/max_area
        else:
            compactness=0

        # Normalizamos el area, circularidad y compacidad
        d0 = 1e-10

        circ_normalized = np.log10(np.abs(circ)+d0)
        compactness_normalized = np.log10(np.abs(compactness)+d0)


        # Rellenamos los contornos

        # Aplicar un umbral binario a la imagen en blanco
        _, thresh_img = cv2.threshold(smooth_img, 128, 255, cv2.THRESH_BINARY)
        copy_img = thresh_img.copy()
        h, w = copy_img.shape[:2]

        # Crear una máscara de ceros del mismo tamaño que la imagen
        mask = np.zeros((h+2, w+2), np.uint8)

        # Rellenar la imagen en blanco con el color blanco                                                      
        cv2.floodFill(copy_img, mask, (0,0), 255)
        
        # Invertir los colores de la imagen en blanco
        inv_img = cv2.bitwise_not(copy_img)
        filled_img = thresh_img | inv_img
        self.filled_img = filled_img
        
        # Calculamos los momentos de la hu

        # Aplicar un umbral binario a la imagen rellenada
        _, bin_img = cv2.threshold(filled_img, 1, 255, cv2.THRESH_BINARY)

        # Calcular los momentos de la imagen
        moments = cv2.moments(bin_img)

        # Calcular los momentos de Hu de la imagen 
        hu_moments = cv2.HuMoments(moments).flatten()

        # Eliminar el 2do, 3ro, 4to, 5to y 7mo momento de Hu
        hu_moments = np.delete(hu_moments, [1,2,3,4,6], axis=0)
        
        # Normalizar los momentos de Hu
        hu_moments_normalized = np.log10(np.abs(hu_moments)+d0)

        # Calculamos el histograma de colores
        hist_size = 256
        hist_range = (0, 256)
        accumulate = False

        # Convertir la imagen de RGB a HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  
        
        # Histograma del canal hue
        hist_hue = cv2.calcHist([hsv], [0], None, [hist_size], hist_range, accumulate=accumulate)
        hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
        
        # Histograma del canal saturacion
        hist_saturation = cv2.calcHist([hsv], [1], None, [hist_size], hist_range, accumulate=accumulate)
        hist_saturation = cv2.normalize(hist_saturation, hist_saturation).flatten()
        
        # # Histograma del canal valor
        # hist_value = cv2.calcHist([hsv], [2], None, [hist_size], hist_range, accumulate=accumulate)
        # hist_value = cv2.normalize(hist_value, hist_value).flatten()

        # # Histograma del canal rojo
        # hist_red = cv2.calcHist([rgb], [0], None, [hist_size], hist_range, accumulate=accumulate)
        # hist_red = cv2.normalize(hist_red, hist_red).flatten()

        # # Histograma del canal verde
        # hist_green = cv2.calcHist([rgb], [1], None, [hist_size], hist_range, accumulate=accumulate)
        # hist_green = cv2.normalize(hist_green, hist_green).flatten()

        # # Histograma del canal azul
        # hist_blue = cv2.calcHist([rgb], [2], None, [hist_size], hist_range, accumulate=accumulate)
        # hist_blue = cv2.normalize(hist_blue, hist_blue).flatten()

        # Concateno los histogramas de colores
        # color_histograms_rgb = np.concatenate([hist_red, hist_green, hist_blue])
        color_histograms_hsv = np.concatenate([hist_hue, hist_saturation])

        # Calculamos el histograma de textura
        # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # lbp = feature.local_binary_pattern(gray, 8, 1, method="uniform")
        # hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
        # hist = hist.astype("float")
        # hist /= hist.sum()

        # Agrego los histogramas de colores a los momentos de Hu
        features = np.concatenate([hu_moments_normalized, color_histograms_hsv, [circ_normalized, compactness_normalized]])
        features = cv2.normalize(features, features).flatten()
        # print(features.shape)

        return features

    def plot_image(self):
            plt.figure(figsize=(5, 5))
            
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # plt.imshow(self.image, cmap='Greys')  
            
            plt.subplot(2, 3, 1)  
            plt.imshow(self.blur, cmap='Greys') 
            plt.title('Filtro mediano') 

            plt.subplot(2, 3, 2)
            plt.imshow(self.sharpened, cmap='Greys')
            plt.title('Sharpened Image')

            plt.subplot(2, 3, 3)
            plt.imshow(self.edges, cmap='Greys')
            plt.title('Deteccion de Bordes')
            
            plt.subplot(2, 3, 4)
            plt.imshow(self.contorned, cmap='Greys')
            plt.title('Contornos')
            
            plt.subplot(2, 3, 5)
            plt.imshow(self.filled_img, cmap='Greys')
            plt.title('Imagenes con Relleno')

            # plt.tight_layout()
            plt.show()

            pass


# test = Image('/home/bruno/fing/IA_I/data/banana/banana2.jpg')
# test.get_features()
# test.plot_image()

test2 = Image('/home/bruno/fing/IA_I/data/banana_sin_fondo/banana2.png')
test2.get_features()
test2.plot_image()

test3 = Image('/home/bruno/fing/IA_I/data/naranja/naranja1.jpg')
test3.get_features()
test3.plot_image()

test4 = Image('/home/bruno/fing/IA_I/data/naranja_sin_fondo/naranja1.jpg')
test4.get_features()
test4.plot_image()

test5 = Image('/home/bruno/fing/IA_I/data/manzana/manzana1.jpg')
test5.get_features()
test5.plot_image()

test6 = Image('/home/bruno/fing/IA_I/data/manzana_sin_fondo/manzana2.png')
test6.get_features()
test6.plot_image()

test7 = Image('/home/bruno/fing/IA_I/data/zapallo/zapallo1.jpg')
test7.get_features()
test7.plot_image()

test8 = Image('/home/bruno/fing/IA_I/data/zapallo_sin_fondo/zapallo2.png')
test8.get_features()
test8.plot_image()

