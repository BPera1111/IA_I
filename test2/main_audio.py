import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import winsound


import k_nn as knn
from audio import Audio

class MainAudio:
    def __init__(self):
        pass

    def classify_dataset(self, carpeta):
        #Clasifico los audios de cada fruta en una lista de objetos distinta
        path = 'D:/Documents/Facultad/4to/IA 1/TF/Proyecto/dataset/audios/filmora'
        path = os.path.join(path, carpeta)

        fruit_list = []
        
        for audio in os.listdir(path):
            fruit = Audio(os.path.join(path, audio))
            fruit_list.append(fruit)

        return fruit_list
    
    def train_model(self, data, labels):
        # Entreno el modelo KNN
        Knn = knn.KNN()
        Knn.learning(data, labels)
        return Knn
    
    def record_audio(self, duration=2, fs=44100):
        # Grabar audio
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        
        folder = 'D:/Documents/Facultad/4to/IA 1/TF/Proyecto/dataset/audios/recorded'

        # Guardar audio con un nombre aleatorio
        i = np.random.randint(0, 100)
        filename = 'recorded_audio' + str(i) + '.wav'
        while os.path.exists(filename):
            i = np.random.randint(0, 100)
            filename = 'recorded_audio' + str(i) + '.wav'

        filename = os.path.join(folder, filename)
        wav.write(filename, fs, audio)
        return filename
    
    def classify_audio(self, Knn, filename):
        # Carpeta de audios de prueba
        folder = 'D:/Documents/Facultad/4to/IA 1/TF/Proyecto/dataset/audios/filmora/test'
        file = np.random.choice(os.listdir(folder))
        path = os.path.join(folder, file)

        # Carpeta donde se encuentra el audio
        # path = os.path.join(filename)

        # Escucho el audio
        winsound.PlaySound(path, winsound.SND_FILENAME)
        
        # Creo un objeto audio
        new_audio = Audio(path)

        # Obtengo las caracteristicas del audio
        features = new_audio.get_features()
        
        #Clasifico el audio
        result = Knn.predict(features)
        
        return result
