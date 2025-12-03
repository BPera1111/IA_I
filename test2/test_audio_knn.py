import os
import sys
import numpy as np

# Importar las clases necesarias
from k_nn import KNN
from audio import Audio


def cargar_dataset(base_path):
    """
    Carga los audios de entrenamiento desde la carpeta audio/data
    """
    frutas = ['banana', 'manzana', 'naranja', 'pera']
    
    features_list = []
    labels_list = []
    
    print("Cargando dataset de entrenamiento...")
    for fruta in frutas:
        fruta_path = os.path.join(base_path, 'data', fruta)
        print(f"  Procesando {fruta}...")
        
        # Listar todos los archivos de audio de la fruta
        archivos = [f for f in os.listdir(fruta_path) if f.endswith('.wav') or f.endswith('.ogg')]
        
        for archivo in archivos:
            audio_path = os.path.join(fruta_path, archivo)
            try:
                # Crear objeto Audio y extraer características
                audio_obj = Audio(audio_path)
                features = audio_obj.get_features()
                
                # Aplanar las características a un vector 1D
                features_flat = features.flatten()
                
                features_list.append(features_flat)
                labels_list.append(fruta)
                
            except Exception as e:
                print(f"    Error procesando {archivo}: {e}")
    
    print(f"\nTotal de audios cargados: {len(features_list)}")
    
    # Convertir a arrays de numpy
    X_train = np.array(features_list)
    y_train = np.array(labels_list)
    
    return X_train, y_train


def probar_audio(knn_model, audio_path, nombre_archivo):
    """
    Prueba un audio con el modelo KNN entrenado
    """
    print(f"\nProbando audio: {nombre_archivo}")
    
    try:
        # Crear objeto Audio y extraer características
        audio_obj = Audio(audio_path)
        features = audio_obj.get_features()
        
        # Aplanar las características
        features_flat = features.flatten()
        
        # Predecir
        prediccion = knn_model.predict(features_flat)
        
        print(f"  Predicción: {prediccion}")
        
        return prediccion
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    # Ruta base del proyecto
    base_path = '/home/bruno/fing/IA_I/audio2'
    
    # 1. Cargar el dataset de entrenamiento
    X_train, y_train = cargar_dataset(base_path)
    
    # 2. Entrenar el modelo KNN
    print("\nEntrenando modelo KNN con k=5...")
    knn_model = KNN(k=5)
    knn_model.learning(X_train, y_train)
    print("Modelo entrenado!")
    
    # 3. Probar con los audios de test
    test_path = os.path.join(base_path, 'test')
    archivos_test = [f for f in os.listdir(test_path) if f.endswith('.ogg') or f.endswith('.wav')]
    
    print("\n" + "="*50)
    print("RESULTADOS DE PRUEBA")
    print("="*50)
    
    resultados = {}
    for archivo in archivos_test:
        audio_path = os.path.join(test_path, archivo)
        prediccion = probar_audio(knn_model, audio_path, archivo)
        
        # Extraer la etiqueta real del nombre del archivo
        etiqueta_real = archivo.replace('_test.ogg', '')
        resultados[archivo] = {
            'prediccion': prediccion,
            'real': etiqueta_real,
            'correcta': prediccion == etiqueta_real
        }
    
    # 4. Mostrar resumen de resultados
    print("\n" + "="*50)
    print("RESUMEN")
    print("="*50)
    
    correctas = sum(1 for r in resultados.values() if r['correcta'])
    total = len(resultados)
    
    for archivo, resultado in resultados.items():
        simbolo = "✓" if resultado['correcta'] else "✗"
        print(f"{simbolo} {archivo}")
        print(f"  Real: {resultado['real']} | Predicción: {resultado['prediccion']}")
    
    print(f"\nPrecisión: {correctas}/{total} ({100*correctas/total:.1f}%)")


if __name__ == "__main__":
    main()
