# INFORME FINAL - PROYECTO DE INTELIGENCIA ARTIFICIAL I

## Sistema Multimodal de Clasificación de Frutas mediante Audio e Imágenes

---

**Estudiante:** Bruno Pera  
**Curso:** Inteligencia Artificial I  
**Fecha:** Diciembre 2025  
**Institución:** Universidad de la República - Facultad de Ingeniería

---

## 1. INTRODUCCIÓN

### 1.1 Objetivo del Proyecto

El objetivo de este proyecto es desarrollar un sistema multimodal de clasificación de frutas que integre dos modalidades de entrada:
1. **Audio**: Reconocimiento de voz para identificar qué fruta se solicita
2. **Imagen**: Clasificación automática de imágenes de frutas mediante clustering no supervisado

El sistema permite al usuario capturar fotos de frutas, pronunciar el nombre de una fruta, y el programa identifica automáticamente cuál de las imágenes capturadas corresponde a la fruta solicitada.

### 1.2 Motivación

La clasificación multimodal de objetos es un problema relevante en múltiples áreas:
- **Asistencia a personas con discapacidad visual**: sistemas que combinan audio y visión
- **Automatización en la industria alimentaria**: clasificación y selección de productos
- **Educación interactiva**: herramientas de aprendizaje para niños

Este proyecto demuestra cómo dos modalidades complementarias (audio e imagen) pueden trabajar juntas para resolver un problema de clasificación de manera robusta.

### 1.3 Descripción General del Sistema

El sistema desarrollado consta de los siguientes componentes principales:

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO DEL SISTEMA                        │
├─────────────────────────────────────────────────────────────┤
│  1. Captura de Imágenes                                     │
│     ├─ Captura 4 fotos con cámara/OBS                      │
│     └─ Remoción automática de fondo (rembg)                │
│                                                             │
│  2. Procesamiento de Audio                                  │
│     ├─ Grabación de voz (barra espaciadora)                │
│     ├─ Extracción de características (RMS, ZCR, derivada)  │
│     └─ Clasificación con KNN (k=5)                         │
│                                                             │
│  3. Procesamiento de Imágenes                               │
│     ├─ Extracción de características (HSV, Hu Moments)     │
│     ├─ Clustering con K-Means (k=4)                        │
│     └─ Asignación de etiquetas por cluster mayoritario     │
│                                                             │
│  4. Integración y Resultado                                 │
│     └─ Identificación y visualización de la imagen         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. MARCO TEÓRICO

### 2.1 Algoritmo K-Nearest Neighbors (KNN)

**Descripción:**  
KNN es un algoritmo de aprendizaje supervisado que clasifica nuevas instancias basándose en la similitud con ejemplos de entrenamiento. 

**Principio de funcionamiento:**
- Dada una nueva muestra $x$, se calculan las distancias a todos los puntos del conjunto de entrenamiento
- Se seleccionan los $k$ vecinos más cercanos
- La clase predicha es la clase mayoritaria entre esos $k$ vecinos

**Distancia euclidiana:**

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**Ventajas:**
- Simple de implementar y entender
- No requiere fase de entrenamiento explícita
- Efectivo para problemas con fronteras de decisión irregulares

**Desventajas:**
- Computacionalmente costoso en predicción (O(n) por consulta)
- Sensible a características irrelevantes y escala de datos
- Requiere selección adecuada de $k$

**Aplicación en el proyecto:**  
Se utiliza KNN con $k=5$ para clasificar audios de frutas en 4 categorías (banana, manzana, naranja, pera).

### 2.2 Algoritmo K-Means

**Descripción:**  
K-Means es un algoritmo de clustering no supervisado que particiona el conjunto de datos en $k$ grupos basándose en la similitud de características.

**Algoritmo:**
1. Inicializar $k$ centroides (aleatorio o K-Means++)
2. Asignar cada punto al centroide más cercano
3. Recalcular centroides como media de puntos asignados
4. Repetir pasos 2-3 hasta convergencia

**Función objetivo:**

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

donde $C_i$ es el cluster $i$ y $\mu_i$ es su centroide.

**K-Means++ (inicialización):**  
Mejora la inicialización aleatoria seleccionando centroides con probabilidad proporcional al cuadrado de su distancia a los centroides ya elegidos.

**Ventajas:**
- Eficiente: O(nki) donde i es el número de iteraciones
- Escalable a grandes datasets
- Converge rápidamente en práctica

**Desventajas:**
- Requiere especificar $k$ de antemano
- Sensible a inicialización (mitigado con K-Means++)
- Asume clusters esféricos y de tamaño similar

**Aplicación en el proyecto:**  
Se utiliza K-Means con $k=4$ para agrupar imágenes de frutas y asignar etiquetas a nuevas fotos basándose en el cluster al que pertenecen.

---

## 3. METODOLOGÍA

### 3.1 Datasets

#### 3.1.1 Dataset de Audio
- **Contenido**: Grabaciones de voz pronunciando nombres de frutas
- **Clases**: 4 (banana, manzana, naranja, pera)
- **Ubicación**: `/audio/data/{fruta}/`
- **Formato**: OGG Vorbis
- **Preprocesamiento**:
  - Recorte de silencios (top_db=30)
  - Preénfasis para realzar altas frecuencias
  - Normalización de amplitud
  - Padding/truncamiento a 1 segundo (44100 muestras)

#### 3.1.2 Dataset de Imágenes
- **Contenido**: Imágenes de frutas sin fondo
- **Clases**: 4 (banana_sin_fondo, manzana_sin_fondo, naranja_sin_fondo, pera_sin_fondo)
- **Ubicación**: `/data_sin_fondo/{fruta_sin_fondo}/`
- **Formato**: PNG con transparencia
- **Preprocesamiento**:
  - Conversión a espacio de color HSV
  - Segmentación por región central y saturación
  - Remoción de fondo con biblioteca rembg

### 3.2 Extracción de Características

#### 3.2.1 Características de Audio

La clase `Audio` extrae 3 tipos de características temporales:

**1. Root Mean Square (RMS)**
- Mide la energía promedio de la señal
- Útil para detectar presencia de voz

**2. Derivada del RMS**
- Cambios en la energía a lo largo del tiempo
- Captura dinámica de la pronunciación

**3. Zero Crossing Rate (ZCR)**
- Frecuencia de cambios de signo en la señal
- Relacionado con contenido espectral

```python
# Transformada de Fourier
tff = librosa.stft(audio)
Spec = np.abs(tff)

# RMS
rms = librosa.feature.rms(S=Spec)

# Derivada
derivate = np.diff(rms, axis=0)

# ZCR
zcr = librosa.feature.zero_crossing_rate(y=audio)

# Vector de características
features = np.concatenate([rms, derivate, zcr])
```

**Dimensionalidad**: Variable (depende de la longitud del audio tras STFT)

#### 3.2.2 Características de Imagen

La clase `FruitImage` extrae 8 características que combinan información de color y forma:

**Características de Color (4 features):**

1. **Hue circular**: $\cos(H)$ y $\sin(H)$
   - Representación circular del matiz para evitar discontinuidad en 0°/360°
   - Captura el color dominante de la fruta

2. **Saturación (S)**: Normalizada [0, 1]
   - Intensidad del color (pureza)

3. **Valor (V)**: Normalizado [0, 1]
   - Brillo de la imagen

**Características de Forma (4 features):**

4-5. **Momentos de Hu (Hu0, Hu1)**:
   - Invariantes a traslación, rotación y escala
   - Normalizados con logaritmo: $\log_{10}(|Hu_i| + \epsilon)$

6. **Circularidad**:
   $$\text{Circularidad} = \frac{4\pi \cdot \text{Área}}{\text{Perímetro}^2}$$
   - Mide qué tan circular es el objeto (1 = círculo perfecto)

7. **Compacidad**:
   $$\text{Compacidad} = \frac{\text{Perímetro}^2}{\text{Área}}$$
   - Relación perímetro-área

**Vector final**: $[cos(H), sin(H), S, V, Hu_0, Hu_1, Circ, Comp]$ ∈ ℝ⁸

### 3.3 Arquitectura del Sistema

El sistema está organizado en módulos independientes:

```
src/
├── audio.py          # Clase Audio: carga y extracción de features
├── k_nn.py           # Implementación de KNN
├── image.py          # Clases FruitImage y FruitDataset
├── kmeans.py         # Implementación de K-Means
└── main.py           # Aplicación principal con menú interactivo
```

#### 3.3.1 Módulo de Audio (`audio.py`)

```python
class Audio:
    def __init__(self, path):
        # Carga, preprocesamiento y normalización
        
    def get_features(self):
        # Extracción de RMS, derivada y ZCR
        return features
```

#### 3.3.2 Módulo KNN (`k_nn.py`)

```python
class KNN:
    def __init__(self, k=5):
        self.k = k
        
    def learning(self, data, labels):
        # Almacena dataset de entrenamiento
        
    def predict(self, test):
        # Calcula distancias, selecciona k vecinos
        # Retorna clase mayoritaria
        
    def euclidean_distance(self, x, y):
        # Distancia euclidiana
```

#### 3.3.3 Módulo de Imagen (`image.py`)

```python
class FruitImage:
    def __init__(self, path):
        # Carga imagen y convierte a HSV
        
    def get_fruit_mask(self):
        # Segmentación: región central + saturación + Hue de fondo
        
    def get_features(self):
        # Extrae 8 características (color + forma)

class FruitDataset:
    def __init__(self, root_dir):
        # Carga dataset completo
        
    def build(self):
        # Procesa todas las imágenes
```

#### 3.3.4 Módulo K-Means (`kmeans.py`)

```python
class SimpleKMeans:
    def __init__(self, k, max_iters=10000, tol=1e-4):
        self.k = k
        
    def _init_centroids(self, X):
        # Inicialización K-Means++
        
    def fit(self, X):
        # Algoritmo iterativo hasta convergencia
        
    def predict(self, X):
        # Asigna muestras a clusters más cercanos
```

#### 3.3.5 Aplicación Principal (`main.py`)

Implementa un menú interactivo con 3 opciones:

**Opción 1: Capturar y procesar fotos**
- Selección de cámara (física o virtual OBS)
- Captura de 4 fotos con visualización en tiempo real
- Remoción automática de fondo con rembg
- Guardado en `/data_sin_fondo/fotos_sin_fondo/`

**Opción 2: Grabar audio y buscar imagen**
1. Carga dataset de audio y entrena KNN
2. Grabación de voz con barra espaciadora
3. Reproducción y clasificación inmediata
4. Confirmación del usuario
5. Carga de todas las imágenes (entrenamiento + capturas)
6. Entrenamiento de K-Means
7. Asignación de etiquetas a fotos capturadas según cluster mayoritario
8. Búsqueda y visualización de la imagen correspondiente

**Opción 3: Salir**

### 3.4 Flujo de Ejecución Detallado

#### Fase 1: Captura de Imágenes
```
Usuario → Menú Opción 1 → Selecciona Cámara → Captura 4 Fotos
    → Remoción de Fondo (rembg) → Guarda PNG con transparencia
```

#### Fase 2: Clasificación por Audio
```
Usuario → Menú Opción 2 → Entrena KNN con dataset existente
    → Graba Voz (mantiene barra espaciadora)
    → Reproduce Audio y Muestra Clasificación
    → Confirma (s/n) → Si no, repite grabación
```

#### Fase 3: Clustering y Etiquetado
```
Carga Imágenes (entrenamiento + capturas) → Extrae Features (8D)
    → Entrena K-Means (k=4) → Asigna cada imagen a un cluster
    → Para cada foto capturada: busca etiqueta mayoritaria en su cluster
```

#### Fase 4: Integración y Resultado
```
Fruta del Audio (ej: "banana") → Busca en fotos capturadas
    → Encuentra imagen con etiqueta "banana" → Visualiza con matplotlib
```

---

## 4. RESULTADOS

### 4.1 Clasificación de Audio (KNN)

**Configuración:**
- Algoritmo: K-Nearest Neighbors
- Valor de k: 5
- Distancia: Euclidiana
- Features: RMS, derivada RMS, ZCR

**Observaciones:**
- El sistema clasifica correctamente audios claros con pronunciación estándar
- La confirmación interactiva permite regrabar si el audio no se capturó bien
- La reproducción inmediata del audio ayuda al usuario a verificar la calidad

### 4.2 Clustering de Imágenes (K-Means)

**Configuración:**
- Algoritmo: K-Means con inicialización K-Means++
- Número de clusters: k=4
- Tolerancia de convergencia: 1e-4
- Features: 8D (color HSV circular + forma)

**Ejemplo de distribución de clusters:**
```
Cluster 0: [banana_sin_fondo, banana_sin_fondo, banana_sin_fondo, foto_1]
Cluster 1: [naranja_sin_fondo, naranja_sin_fondo, naranja_sin_fondo, foto_2]
Cluster 2: [manzana_sin_fondo, manzana_sin_fondo, manzana_sin_fondo, foto_3]
Cluster 3: [pera_sin_fondo, pera_sin_fondo, pera_sin_fondo, foto_4]
```

**Asignación de etiquetas:**
Cada foto capturada hereda la etiqueta mayoritaria de su cluster:
- `foto_1.png` → "banana" (cluster 0 dominado por bananas)
- `foto_2.png` → "naranja" (cluster 1 dominado por naranjas)
- Y así sucesivamente

### 4.3 Integración Multimodal

**Escenario de uso típico:**
1. Usuario captura 4 fotos: banana, manzana, naranja, pera
2. Usuario pronuncia "banana"
3. Sistema:
   - Clasifica audio → "banana"
   - Busca en fotos capturadas → encuentra `foto_1.png` con etiqueta "banana"
   - Muestra la imagen correcta

**Ventajas del enfoque:**
- **Robustez**: Si una modalidad falla, la otra puede compensar
- **Flexibilidad**: Usuario puede capturar fotos en cualquier orden
- **Interactividad**: Confirmación en cada paso
- **Extensibilidad**: Fácil agregar más frutas o modalidades

### 4.4 Visualización de Resultados

El sistema muestra:
- **Durante captura**: Preview en tiempo real con OpenCV
- **Durante audio**: Mensaje de clasificación inmediata
- **Resultado final**: Imagen con matplotlib (8x8 pulgadas) mostrando:
  - Fruta solicitada (del audio)
  - Nombre de archivo
  - Mensaje de confirmación

---

## 5. ANÁLISIS Y DISCUSIÓN

### 5.1 Fortalezas del Sistema

**1. Modularidad**
- Código organizado en clases independientes
- Fácil mantenimiento y extensión
- Cada módulo testeable por separado

**2. Robustez en Preprocesamiento**
- Audio: normalización, padding, filtrado
- Imagen: múltiples estrategias de segmentación (región central, saturación, Hue de fondo)

**3. Experiencia de Usuario**
- Menú interactivo para múltiples sesiones
- Confirmación en cada paso crítico
- Visualización en tiempo real

**4. Flexibilidad de Hardware**
- Soporte para múltiples cámaras (físicas o virtuales como OBS)
- Compatible con diferentes dispositivos de audio

### 5.2 Limitaciones y Desafíos

**1. Dependencia de Calidad de Datos**
- Audio: ruido ambiental puede afectar clasificación
- Imagen: iluminación y ángulo de captura son críticos

**2. Escalabilidad**
- KNN: O(n) en predicción, lento con datasets grandes
- K-Means: requiere especificar k manualmente

**3. Generalización**
- Sistema entrenado con acentos/pronunciaciones específicas
- Frutas con colores similares pueden confundirse

**4. Asignación por Cluster Mayoritario**
- Si un cluster contiene mezcla de frutas, la asignación puede ser incorrecta
- Asume clusters bien separados

### 5.3 Posibles Mejoras

**Corto Plazo:**
1. **Validación de clusters**: calcular métricas como Silhouette Score
2. **Selección automática de k**: usar Elbow Method o Gap Statistic
3. **Data augmentation**: aumentar dataset de audio y imágenes
4. **Configuración persistente**: guardar índices de cámara/micrófono

**Largo Plazo:**
1. **Deep Learning**:
   - Audio: CNN 1D o RNN para clasificación
   - Imagen: Transfer Learning con ResNet/EfficientNet
2. **Fusión multimodal**: combinar scores de ambas modalidades con pesos aprendibles
3. **Detección de anomalías**: identificar frutas desconocidas
4. **Interfaz gráfica**: GUI con PyQt o Tkinter
5. **Despliegue web**: API REST con Flask/FastAPI

---

## 6. CONCLUSIONES

Este proyecto demuestra exitosamente la integración de dos modalidades (audio e imagen) para resolver un problema de clasificación de frutas. Los principales logros son:

1. **Implementación desde cero** de algoritmos clásicos (KNN y K-Means) sin usar bibliotecas de alto nivel
2. **Sistema funcional end-to-end** con captura, procesamiento, clasificación y visualización
3. **Buena arquitectura de software** con separación de responsabilidades y reutilización de código
4. **Experiencia de usuario interactiva** con menú, confirmaciones y feedback inmediato

**Lecciones aprendidas:**
- La calidad del preprocesamiento es crucial para el rendimiento
- La ingeniería de características requiere conocimiento del dominio
- Los algoritmos clásicos siguen siendo efectivos para problemas acotados
- La multimodalidad aumenta la robustez del sistema

**Aplicabilidad:**
Los conceptos y técnicas de este proyecto son aplicables a:
- Sistemas de asistencia para personas con discapacidad
- Automatización en logística y retail
- Herramientas educativas interactivas
- Prototipos de IoT y robótica

---

## 7. REFERENCIAS

### Bibliotecas y Frameworks
- **NumPy**: Harris, C.R., et al. (2020). Array programming with NumPy. Nature, 585, 357–362.
- **OpenCV**: Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
- **Librosa**: McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. SciPy.
- **rembg**: Qin, X., et al. (2020). U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection.

### Algoritmos
- **K-Means**: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- **K-Means++**: Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding.
- **KNN**: Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory.
- **Hu Moments**: Hu, M.K. (1962). Visual pattern recognition by moment invariants. IRE Transactions on Information Theory.

### Recursos Adicionales
- Documentación oficial de scikit-learn (referencia conceptual)
- Tutorial de procesamiento de audio digital (DSP)
- OpenCV documentation para procesamiento de imágenes

---

## ANEXOS

### A. Estructura del Proyecto

```
IA_I/
├── src/                      # Código fuente principal
│   ├── audio.py              # Clase Audio
│   ├── k_nn.py               # Implementación KNN
│   ├── image.py              # Clases FruitImage y FruitDataset
│   ├── kmeans.py             # Implementación K-Means
│   └── main.py               # Aplicación principal
├── audio/                    # Datasets de audio
│   ├── data/
│   │   ├── banana/
│   │   ├── manzana/
│   │   ├── naranja/
│   │   └── pera/
│   └── test/
├── data_sin_fondo/           # Datasets de imágenes
│   ├── banana_sin_fondo/
│   ├── manzana_sin_fondo/
│   ├── naranja_sin_fondo/
│   ├── pera_sin_fondo/
│   └── fotos_sin_fondo/      # Fotos capturadas por el usuario
├── INFORME_FINAL.md          # Este documento
└── README.md                 # Instrucciones de uso
```

### B. Requisitos del Sistema

**Software:**
- Python 3.10+
- OpenCV 4.x con soporte GUI (libgtk2.0-dev)
- PortAudio (para sounddevice)

**Hardware mínimo:**
- Cámara web (física o virtual con OBS)
- Micrófono
- 4 GB RAM
- Procesador dual-core

**Dependencias Python:**
```
numpy>=1.21.0
opencv-python>=4.5.0
librosa>=0.9.0
sounddevice>=0.4.0
scipy>=1.7.0
matplotlib>=3.4.0
pillow>=8.0.0
rembg>=2.0.0
pynput>=1.7.0
```

### C. Instrucciones de Instalación y Uso

**1. Clonar repositorio:**
```bash
git clone https://github.com/BPera1111/IA_I.git
cd IA_I
```

**2. Crear entorno virtual:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Instalar dependencias del sistema (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libgtk2.0-dev pkg-config portaudio19-dev
```

**4. Instalar dependencias Python:**
```bash
pip install numpy opencv-python librosa sounddevice scipy matplotlib pillow rembg pynput
```

**5. Ejecutar aplicación:**
```bash
cd src
python main.py
```

**6. Uso interactivo:**
- **Opción 1**: Capturar 4 fotos (presiona ESPACIO para cada una)
- **Opción 2**: Grabar audio (mantén BARRA ESPACIADORA), confirma y ve el resultado
- **Opción 3**: Salir

### D. Configuración de OBS (opcional)

Para usar OBS Virtual Camera:
1. En OBS: `Tools → Start Virtual Camera`
2. En el programa: selecciona el índice de cámara correspondiente

Para audio de OBS (Linux):
1. Configurar un sink virtual en PulseAudio/PipeWire
2. En OBS: `Edit → Advanced Audio Properties → Audio Monitoring: Monitor and Output`
3. Seleccionar el dispositivo "Monitor of..." en el programa

---

## AGRADECIMIENTOS

Agradezco al curso de Inteligencia Artificial I de la Facultad de Ingeniería por proporcionar los conocimientos teóricos necesarios para este proyecto, y a la comunidad open-source por las excelentes bibliotecas utilizadas.

---

**Fin del Informe**

*Este documento describe el desarrollo completo del sistema multimodal de clasificación de frutas, desde la fundamentación teórica hasta la implementación práctica y resultados obtenidos.*
