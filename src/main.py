"""
Programa principal que integra clasificación de audio y clustering de imágenes.

Flujo:
1. Graba un audio de 2 segundos
2. Clasifica el audio con KNN para identificar la fruta
3. Carga el dataset de imágenes (incluye test_sin_fondo)
4. Entrena K-Means con todas las imágenes
5. Identifica qué imagen de test_sin_fondo corresponde a la fruta del audio
6. Muestra la imagen identificada
"""

import os
import sys
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from pynput import keyboard
import time
from PIL import Image
from rembg import remove

from audio import Audio
from k_nn import KNN
from image import FruitImage, FruitDataset
from kmeans import SimpleKMeans


def listar_camaras_disponibles():
    """
    Lista todas las cámaras disponibles en el sistema
    """
    print("Buscando cámaras disponibles...")
    camaras = []
    
    # Probar los primeros 10 índices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Obtener información de la cámara
                backend = cap.getBackendName()
                camaras.append((i, backend))
                print(f"  [{i}] Cámara detectada (Backend: {backend})")
            cap.release()
    
    return camaras


def seleccionar_camara():
    """
    Permite al usuario seleccionar la cámara a usar
    """
    print("="*60)
    print("SELECCIÓN DE CÁMARA")
    print("="*60)
    
    camaras = listar_camaras_disponibles()
    
    if not camaras:
        print("\n⚠ No se detectaron cámaras disponibles")
        return 0
    
    print(f"\nSe detectaron {len(camaras)} cámara(s)")
    print("\nPara usar OBS Virtual Camera:")
    print("  - Asegúrate de que OBS esté corriendo")
    print("  - Ve a Tools → Start Virtual Camera en OBS")
    print("  - La cámara virtual suele aparecer con un índice mayor (ej: 2, 4, etc.)\n")
    
    while True:
        try:
            indice = input(f"Ingresa el número de cámara a usar [0-9] (default=0): ").strip()
            if indice == "":
                return 0
            indice = int(indice)
            if 0 <= indice <= 9:
                return indice
            else:
                print("Por favor ingresa un número entre 0 y 9")
        except ValueError:
            print("Por favor ingresa un número válido")


def capturar_fotos(output_dir, num_fotos=4, camera_index=0):
    """
    Captura 4 fotos con la cámara y las guarda temporalmente
    """
    print("="*60)
    print("CAPTURA DE FOTOS")
    print("="*60)
    print(f"Vamos a capturar {num_fotos} fotos")
    print("Presiona ESPACIO para capturar cada foto")
    print("Presiona 'q' para salir\n")
    
    # Crear carpeta temporal para fotos originales
    temp_dir = '/tmp/fotos_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Iniciar cámara con el índice seleccionado
    print(f"Intentando abrir cámara con índice {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("⚠ Error: No se pudo abrir la cámara")
        return []
    
    fotos_capturadas = []
    foto_num = 1
    
    print(f"Preparando foto {foto_num}/{num_fotos}...")
    
    while foto_num <= num_fotos:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠ Error al capturar frame")
            break
        
        # Mostrar preview
        cv2.putText(frame, f"Foto {foto_num}/{num_fotos} - Presiona ESPACIO para capturar", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Captura de Fotos', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espacio
            # Guardar foto
            foto_path = os.path.join(temp_dir, f'foto_{foto_num}.jpg')
            cv2.imwrite(foto_path, frame)
            fotos_capturadas.append(foto_path)
            
            print(f"✓ Foto {foto_num}/{num_fotos} capturada")
            foto_num += 1
            
            if foto_num <= num_fotos:
                print(f"Preparando foto {foto_num}/{num_fotos}...")
        
        elif key == ord('q'):  # Salir
            print("\n⚠ Captura cancelada")
            break
    
    # Liberar cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
    
    if len(fotos_capturadas) == num_fotos:
        print(f"\n✓ {num_fotos} fotos capturadas exitosamente\n")
    else:
        print(f"\n⚠ Solo se capturaron {len(fotos_capturadas)}/{num_fotos} fotos\n")
    
    return fotos_capturadas


def remover_fondo_fotos(fotos_paths, output_dir):
    """
    Remueve el fondo de las fotos capturadas usando rembg
    """
    print("="*60)
    print("REMOVIENDO FONDO DE LAS FOTOS")
    print("="*60)
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    fotos_sin_fondo = []
    
    for i, foto_path in enumerate(fotos_paths, 1):
        try:
            print(f"Procesando foto {i}/{len(fotos_paths)}...", end=" ")
            
            # Abrir imagen y remover fondo
            input_img = Image.open(foto_path)
            output_img = remove(input_img)
            
            # Guardar como PNG (para soportar transparencia)
            output_path = os.path.join(output_dir, f'foto_{i}.png')
            output_img.save(output_path, 'PNG')
            
            fotos_sin_fondo.append(output_path)
            print(f"✓")
            
        except Exception as e:
            print(f"⚠ Error: {e}")
    
    print(f"\n✓ {len(fotos_sin_fondo)} fotos procesadas y guardadas en {output_dir}\n")
    
    return fotos_sin_fondo


def cargar_dataset_audio(base_path):
    """
    Carga los audios de entrenamiento desde audio/data
    """
    frutas = ['banana', 'manzana', 'naranja', 'pera']
    
    features_list = []
    labels_list = []
    
    print("="*60)
    print("CARGANDO DATASET DE AUDIO")
    print("="*60)
    
    for fruta in frutas:
        fruta_path = os.path.join(base_path, 'data', fruta)
        print(f"  Procesando {fruta}...")
        
        archivos = [f for f in os.listdir(fruta_path) if f.endswith('.ogg')]
        
        for archivo in archivos:
            audio_path = os.path.join(fruta_path, archivo)
            try:
                audio_obj = Audio(audio_path)
                features = audio_obj.get_features()
                features_flat = features.flatten()
                
                features_list.append(features_flat)
                labels_list.append(fruta)
                
            except Exception as e:
                print(f"    Error procesando {archivo}: {e}")
    
    print(f"Total de audios cargados: {len(features_list)}\n")
    
    X_train = np.array(features_list)
    y_train = np.array(labels_list)
    
    return X_train, y_train


def grabar_audio(fs=44100):
    """
    Graba audio mientras se mantiene presionada la barra espaciadora
    """
    print("="*60)
    print("GRABACIÓN DE AUDIO")
    print("="*60)
    print("Mantén presionada la BARRA ESPACIADORA para grabar")
    print("Suelta la tecla para terminar la grabación")
    print("Esperando...")
    
    grabando = False
    audio_chunks = []
    
    def on_press(key):
        nonlocal grabando
        if key == keyboard.Key.space and not grabando:
            grabando = True
            print("\n🔴 GRABANDO... (suelta la barra para terminar)")
            # Iniciar grabación en streaming
            audio_chunks.clear()
    
    def on_release(key):
        nonlocal grabando
        if key == keyboard.Key.space and grabando:
            grabando = False
            print("✓ Grabación completada\n")
            return False  # Detener el listener
    
    # Callback para capturar audio en tiempo real
    def audio_callback(indata, frames, time, status):
        if grabando:
            audio_chunks.append(indata.copy())
    
    # Iniciar stream de audio
    stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback)
    stream.start()
    
    # Listener de teclado
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    
    # Detener stream
    stream.stop()
    stream.close()
    
    # Concatenar chunks de audio
    if len(audio_chunks) > 0:
        audio = np.concatenate(audio_chunks, axis=0)
    else:
        print("⚠ No se grabó audio. Intentando de nuevo...")
        return grabar_audio(fs)
    
    # Guardar audio temporalmente
    temp_dir = '/tmp'
    filename = os.path.join(temp_dir, 'recorded_audio.wav')
    wav.write(filename, fs, audio)
    
    return filename


def reproducir_audio(audio_path, fs=44100):
    """
    Reproduce el audio grabado para verificación
    """
    print("="*60)
    print("REPRODUCCIÓN DEL AUDIO")
    print("="*60)
    print("Reproduciendo audio en 2 segundos...")
    time.sleep(2)
    
    try:
        # Leer el archivo de audio
        _, audio_data = wav.read(audio_path)
        
        # Reproducir
        print("🔊 Reproduciendo...")
        sd.play(audio_data, fs)
        sd.wait()
        print("✓ Reproducción completada\n")
        
    except Exception as e:
        print(f"Error al reproducir: {e}\n")


def confirmar_grabacion():
    """
    Pregunta al usuario si la grabación fue correcta
    """
    print("="*60)
    print("CONFIRMACIÓN")
    print("="*60)
    
    while True:
        respuesta = input("¿El audio se grabó correctamente? (s/n): ").strip().lower()
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            print("✓ Continuando con la clasificación...\n")
            return True
        elif respuesta in ['n', 'no']:
            print("⟳ Preparando nueva grabación...\n")
            return False
        else:
            print("Por favor, responde 's' para sí o 'n' para no")


def clasificar_audio(knn_model, audio_path):
    """
    Clasifica un audio grabado usando el modelo KNN
    """
    print("="*60)
    print("CLASIFICACIÓN DE AUDIO")
    print("="*60)
    
    try:
        audio_obj = Audio(audio_path)
        features = audio_obj.get_features()
        features_flat = features.flatten()
        
        prediccion = knn_model.predict(features_flat)
        
        print(f"✓ Fruta detectada: {prediccion.upper()}\n")
        
        return prediccion
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def cargar_dataset_imagenes(base_path):
    """
    Carga todas las imágenes (entrenamiento + fotos_sin_fondo)
    """
    print("="*60)
    print("CARGANDO DATASET DE IMÁGENES")
    print("="*60)
    
    ds = FruitDataset(base_path)
    ds.build(sat_threshold=40, central_fraction=0.6)
    
    X, y = ds.get_data()
    image_paths = ds.get_image_paths()
    
    print(f"Total de imágenes cargadas: {len(X)}\n")
    
    return X, y, image_paths


def entrenar_kmeans(X, k=4):
    """
    Entrena K-Means con las imágenes
    """
    print("="*60)
    print("ENTRENANDO K-MEANS")
    print("="*60)
    
    kmeans = SimpleKMeans(k=k, max_iters=100, tol=1e-4, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    
    print("✓ Entrenamiento completado\n")
    
    return kmeans, labels


def asignar_etiquetas_test(labels, y_dataset, image_paths):
    """
    Para cada imagen de fotos_sin_fondo, encuentra su cluster y
    le asigna la etiqueta mayoritaria de ese cluster.
    
    Retorna un diccionario: {ruta_imagen_test: etiqueta_asignada}
    """
    print("="*60)
    print("ASIGNANDO ETIQUETAS A FOTOS CAPTURADAS")
    print("="*60)
    
    test_labels = {}
    
    # Encontrar índices de imágenes de fotos_sin_fondo
    test_indices = [i for i, path in enumerate(image_paths) if 'fotos_sin_fondo' in path]
    
    for idx in test_indices:
        cluster_id = labels[idx]
        img_path = image_paths[idx]
        
        # Encontrar todas las imágenes en el mismo cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # Obtener etiquetas de las imágenes del cluster (excluyendo fotos_sin_fondo)
        cluster_labels = [y_dataset[i] for i in cluster_indices if 'fotos_sin_fondo' not in image_paths[i]]
        
        if cluster_labels:
            # Etiqueta mayoritaria
            etiqueta_mayoritaria = Counter(cluster_labels).most_common(1)[0][0]
            # Limpiar nombre (quitar "_sin_fondo")
            etiqueta_limpia = etiqueta_mayoritaria.replace('_sin_fondo', '')
            
            test_labels[img_path] = etiqueta_limpia
            
            nombre_archivo = os.path.basename(img_path)
            print(f"  {nombre_archivo} → Cluster {cluster_id} → {etiqueta_limpia}")
    
    print()
    return test_labels


def mostrar_imagen(img_path, fruta_predicha):
    """
    Carga todas las imágenes (entrenamiento + test_sin_fondo)
    """
    print("="*60)
    print("CARGANDO DATASET DE IMÁGENES")
    print("="*60)
    
    ds = FruitDataset(base_path)
    ds.build(sat_threshold=40, central_fraction=0.6)
    
    X, y = ds.get_data()
    image_paths = ds.get_image_paths()
    
    print(f"Total de imágenes cargadas: {len(X)}\n")
    
    return X, y, image_paths


def entrenar_kmeans(X, k=4):
    """
    Entrena K-Means con las imágenes
    """
    print("="*60)
    print("ENTRENANDO K-MEANS")
    print("="*60)
    
    kmeans = SimpleKMeans(k=k, max_iters=100, tol=1e-4, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    
    print("✓ Entrenamiento completado\n")
    
    return kmeans, labels


def asignar_etiquetas_test(labels, y_dataset, image_paths):
    """
    Para cada imagen de test_sin_fondo, encuentra su cluster y
    le asigna la etiqueta mayoritaria de ese cluster.
    
    Retorna un diccionario: {ruta_imagen_test: etiqueta_asignada}
    """
    print("="*60)
    print("ASIGNANDO ETIQUETAS A IMÁGENES DE FOTOS SIN FONDO")
    print("="*60)
    
    test_labels = {}
    
    # Encontrar índices de imágenes de fotos_sin_fondo
    test_indices = [i for i, path in enumerate(image_paths) if 'fotos_sin_fondo' in path]
    
    for idx in test_indices:
        cluster_id = labels[idx]
        img_path = image_paths[idx]
        
        # Encontrar todas las imágenes en el mismo cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # Obtener etiquetas de las imágenes del cluster (excluyendo fotos_sin_fondo)
        cluster_labels = [y_dataset[i] for i in cluster_indices if 'fotos_sin_fondo' not in image_paths[i]]
        
        if cluster_labels:
            # Etiqueta mayoritaria
            etiqueta_mayoritaria = Counter(cluster_labels).most_common(1)[0][0]
            # Limpiar nombre (quitar "_sin_fondo")
            etiqueta_limpia = etiqueta_mayoritaria.replace('_sin_fondo', '')
            
            test_labels[img_path] = etiqueta_limpia
            
            nombre_archivo = os.path.basename(img_path)
            print(f"  {nombre_archivo} → Cluster {cluster_id} → {etiqueta_limpia}")
    
    print()
    return test_labels


def mostrar_imagen(img_path, fruta_predicha):
    """
    Muestra la imagen identificada
    """
    print("="*60)
    print("RESULTADO FINAL")
    print("="*60)
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img is not None:
        # Convertir a RGB para matplotlib
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # BGRA
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:  # BGR
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # Escala de grises
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        nombre_archivo = os.path.basename(img_path)
        
        print(f"Fruta solicitada (audio): {fruta_predicha.upper()}")
        print(f"Imagen identificada: {nombre_archivo}")
        print(f"✓ ¡COINCIDENCIA ENCONTRADA!\n")
        
        # Mostrar imagen
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.title(f"Fruta: {fruta_predicha.upper()}\nArchivo: {nombre_archivo}", 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Error: No se pudo cargar la imagen {img_path}")


def main():
    # Rutas base
    audio_base_path = '/home/bruno/fing/IA_I/audio'
    imagen_base_path = '/home/bruno/fing/IA_I/data_sin_fondo'
    fotos_sin_fondo_dir = os.path.join(imagen_base_path, 'fotos_sin_fondo')
    
    # ============================================================
    # PARTE 0: CAPTURAR Y PROCESAR FOTOS
    # ============================================================
    
    # 1. Seleccionar cámara
    camera_index = seleccionar_camara()
    
    # 2. Capturar 4 fotos
    fotos_capturadas = capturar_fotos(fotos_sin_fondo_dir, num_fotos=4, camera_index=camera_index)
    
    if len(fotos_capturadas) < 4:
        print("⚠ No se capturaron suficientes fotos. Abortando.")
        return
    
    # 3. Remover fondo de las fotos
    fotos_sin_fondo = remover_fondo_fotos(fotos_capturadas, fotos_sin_fondo_dir)
    
    if len(fotos_sin_fondo) < 4:
        print("⚠ Error al procesar las fotos. Abortando.")
        return
    
    # ============================================================
    # PARTE 1: AUDIO
    # ============================================================
    
    # 4. Cargar dataset de audio
    X_audio, y_audio = cargar_dataset_audio(audio_base_path)
    
    # 5. Entrenar modelo KNN
    print("="*60)
    print("ENTRENANDO MODELO KNN (AUDIO)")
    print("="*60)
    knn_model = KNN(k=5)
    knn_model.learning(X_audio, y_audio)
    print("✓ Modelo KNN entrenado\n")
    
    # 6. Grabar audio (con reintentos si no está bien)
    audio_grabado = None
    fruta_predicha = None
    while audio_grabado is None:
        audio_temp = grabar_audio()
        
        # Reproducir para verificar
        reproducir_audio(audio_temp)
        
        # Clasificar provisionalmente para mostrar al usuario
        fruta_temp = clasificar_audio(knn_model, audio_temp)
        
        # Confirmar si está bien
        if confirmar_grabacion():
            audio_grabado = audio_temp
            fruta_predicha = fruta_temp
        else:
            print("Intenta de nuevo...\n")
    
    if fruta_predicha is None:
        print("Error al clasificar el audio. Abortando.")
        return
    
    # ============================================================
    # PARTE 2: IMÁGENES
    # ============================================================
    
    # 7. Cargar dataset de imágenes (incluye fotos_sin_fondo)
    X_img, y_img, image_paths = cargar_dataset_imagenes(imagen_base_path)
    
    # 8. Entrenar K-Means
    kmeans_model, labels = entrenar_kmeans(X_img, k=4)
    
    # 9. Asignar etiquetas a fotos capturadas
    test_labels = asignar_etiquetas_test(labels, y_img, image_paths)
    
    # 10. Buscar la imagen que corresponde a la fruta predicha
    imagen_encontrada = None
    for img_path, etiqueta in test_labels.items():
        if etiqueta == fruta_predicha:
            imagen_encontrada = img_path
            break
    
    if imagen_encontrada:
        mostrar_imagen(imagen_encontrada, fruta_predicha)
    else:
        print("="*60)
        print("⚠ ADVERTENCIA")
        print("="*60)
        print(f"No se encontró una imagen de test para la fruta: {fruta_predicha}")
        print(f"Etiquetas disponibles en test: {list(test_labels.values())}")


if __name__ == "__main__":
    main()
