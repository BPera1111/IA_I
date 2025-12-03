import numpy as np
import librosa
import librosa.display

class Audio:
    def __init__(self, path):
        # Cargar del audio
        self.path = path
        audio, self.sr = librosa.load(path=self.path, sr=44100, mono=True)

        # Recortar el audio
        trimmed, _ = librosa.effects.trim(y=audio, top_db=30)
        audio = trimmed
    
        # Filtro el audio y lo normalizo
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)
        
        # Rellenar o truncar para que tenga la misma longitud (se rellena con ceros)
        target_length = int(self.sr * 1)
        if len(audio) > target_length:
            # Si el audio es mas largo que la longitud objetivo, lo recorto
            audio = audio[:target_length]
        else:
            # Si el audio es mas corto que la longitud objetivo, lo relleno con ceros
            padded_audio = np.zeros(target_length)
            padded_audio[:len(audio)] = audio
            audio = padded_audio

        self.audio = audio

    def get_features(self):

        # Obtencion de las caracteristicas
        # Calculo de la transformada de fourier
        tff = librosa.stft(self.audio)

        # Calculo de la magnitud
        Spec = np.abs(tff)

        # Calculo de la raiz de la media cuadratica
        rms = librosa.feature.rms(S=Spec, pad_mode='constant')

        # Calculo de la derivada
        derivate = np.concatenate([np.zeros((1, rms.shape[1])), np.diff(rms, axis=0)], axis=0)

        # Calculo de la tasa de cruces por cero
        zcr = librosa.feature.zero_crossing_rate(y=self.audio)

        # Concatenacion de las caracteristicas
        self.features = np.concatenate([rms, derivate, zcr], axis=0)
        
        return self.features
