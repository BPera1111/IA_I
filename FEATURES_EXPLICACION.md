# Explicación de Features de Clasificación de Frutas

Este documento explica en detalle cómo se extraen y qué representan los features de cada imagen en el sistema de clasificación de frutas.

---

## 1. Composición del Vector de Features

Cada imagen genera un vector de **516 valores** que se dividen en 5 categorías principales:

### **Total: 2 (Hu Moments) + 256 (Hue) + 256 (Saturación) + 1 (Circularidad) + 1 (Compacidad) = 516 valores**

---

## 2. Momentos de Hu (2 valores)

### ¿Qué son?

Los **Hu Moments** son descriptores matemáticos que capturan características de forma de un objeto, con la propiedad especial de ser **invariantes a rotación, escala y traslación**.

### Cálculo en el código:

```python
moments = cv2.moments(bin_img)           # Calcula momentos del objeto
hu_moments = cv2.HuMoments(moments).flatten()  # Extrae los 7 Hu Moments

# Se eliminan algunos momentos (quedan solo 2)
hu_moments = np.delete(hu_moments, [1,2,3,4,6], axis=0)

# Se normalizan con logaritmo
hu_moments_normalized = np.log10(np.abs(hu_moments)+d0)
```

### Los 7 Momentos de Hu (teoría):

1. **hu[0]** - Captura la forma general y tamaño relativo
2. **hu[1]** - Simetría y alargamiento
3. **hu[2]** - Características de forma asimétrica
4. **hu[3]** - Diferencias en distribución de masa
5. **hu[4]** - Orientación y elongación
6. **hu[5]** - Asimetría diagonal
7. **hu[6]** - Asimetría rotacional

### En este proyecto:

Se utilizan solo **2 de los 7 momentos** (hu[0] y hu[3]) porque son los que mejor distinguen entre las frutas.

### Propiedades clave:

✅ **Invariantes a**: rotación, escala, traslación y reflexión  
✅ **Robustos**: funcionan incluso con pequeñas variaciones  
✅ **Rápidos**: cálculo computacional eficiente  

### Ejemplo visual:

- Una **banana alargada** vs una **naranja redonda** → hu_moments diferentes
- Aunque rotes la banana → hu_moments iguales (¡por eso son invariantes!)

---

## 3. Histograma Hue (256 valores)

### ¿Qué es?

El **Hue** es uno de los 3 canales del espacio de color HSV. Representa el **color puro** de cada píxel, independiente del brillo o saturación.

### Escala de Hue:

```
0°      = Rojo
30°     = Naranja
60°     = Amarillo
90°     = Verde
120°    = Cian
150°    = Azul
180°    = Magenta/Violeta
```

### Cálculo en el código:

```python
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # Convierte a HSV
hist_hue = cv2.calcHist([hsv], [0], None, [256], (0, 256))
hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
```

**Resultado**: Un vector de **256 valores** donde cada posición representa la frecuencia de ese color en la imagen (normalizado entre 0 y 1).

### Ejemplo práctico con frutas:

```
Banana:
  - Picos altos en 30-60° (amarillo-naranja)
  - Histograma concentrado en esa zona

Naranja:
  - Picos altos en 0-30° (naranja-rojo)
  - Histograma concentrado en esa zona

Manzana:
  - Picos altos en 0° o 120° (rojo/verde según variedad)
  - Depende del tipo de manzana

Zapallo:
  - Picos altos en 30-60° (naranja)
  - Similar a banana, pero con forma diferente
```

### Visualización conceptual:

```
Histograma Hue - Banana:
  Frecuencia
      |     ╱╲
      |    ╱  ╲
      |   ╱    ╲
      |__╱______╲___
        0  60  120 180
        ↑
      Amarillo-Naranja
```

### ¿Por qué es útil?

✅ Distingue frutas por color (naranja vs banana vs manzana)  
✅ Independiente del brillo (funciona si está oscuro o claro)  
✅ Robusto al tamaño de la fruta  

---

## 4. Histograma Saturación (256 valores)

### ¿Qué es?

La **Saturación** es otro canal del HSV. Mide **qué tan vivo/intenso es el color** (0 = gris/desaturado, 255 = color puro/vivo).

### Cálculo en el código:

```python
hist_saturation = cv2.calcHist([hsv], [1], None, [256], (0, 256))
hist_saturation = cv2.normalize(hist_saturation, hist_saturation).flatten()
```

**Resultado**: Un vector de **256 valores** donde cada posición representa la frecuencia de ese nivel de saturación.

### ¿Por qué necesitamos ambos (Hue + Saturación)?

```
Banana amarilla viva:
  - Hue: 60° (amarillo)
  - Saturación: ALTA (color vivo y puro)

Banana amarilla pálida/madura:
  - Hue: 60° (sigue siendo amarillo)
  - Saturación: BAJA (color más grisáceo/apagado)
```

Con **solo Hue**, ambas se verían iguales.  
Con **Hue + Saturación**, puedes distinguir el estado de madurez/calidad.

### Ejemplos con frutas:

```
Naranja fresca:
  - Hue: naranja
  - Saturación: ALTA (color brillante)

Naranja podrida:
  - Hue: naranja
  - Saturación: BAJA (color apagado)

Manzana roja brillante:
  - Hue: rojo
  - Saturación: ALTA

Manzana roja oscura:
  - Hue: rojo
  - Saturación: MEDIA/BAJA
```

### Resumen:

```
Hue + Saturación juntos te dan:
  ✅ Color base (naranja vs banana)
  ✅ Intensidad/vivacidad del color
  ✅ Detectar variaciones dentro del mismo tipo
```

Sin Saturación, perderías información sobre el "brillo" del color, lo que podría confundir frutas de diferentes estados de madurez.

---

## 5. Circularidad (1 valor)

### ¿Qué es?

La **circularidad** mide **qué tan redonda/circular es una forma**.

### Fórmula:

```
circularidad = (4 × π × área) / (perímetro²)
```

### Rango de valores:

- **Circularidad = 1.0** → Forma perfectamente circular/esférica
- **Circularidad < 1.0** → Forma más irregular/alargada
- Cuanto más bajo → más irregular y alargada

### Cálculo en el código:

```python
per = cv2.arcLength(smooth, True)        # Perímetro del contorno
circ = 4*np.pi*(max_area/(per*per))      # Fórmula de circularidad

circ_normalized = np.log10(np.abs(circ)+d0)  # Se normaliza con logaritmo
```

### Ejemplos con tus frutas:

```
Naranja:
  - Forma redonda
  - circularidad ≈ 0.95-1.0 (MÁS CIRCULAR)

Banana:
  - Forma alargada
  - circularidad ≈ 0.3-0.5 (MENOS CIRCULAR)

Manzana:
  - Forma semi-redonda
  - circularidad ≈ 0.7-0.9 (MEDIA)

Zapallo:
  - Forma irregular/achatada
  - circularidad ≈ 0.4-0.6 (POCO CIRCULAR)
```

### ¿Por qué se normaliza con log10?

Porque los valores de circularidad están entre 0 y 1, y el logaritmo los expande para mejor discriminación:

```
log10(1.0)  = 0
log10(0.5)  ≈ -0.3
log10(0.1)  = -1
```

### ¿Por qué es útil?

✅ **Distingue formas**: banana alargada vs naranja redonda  
✅ **Robusto**: no importa el tamaño, la proporción es la misma  
✅ **Discriminante**: es uno de los mejores features para clasificar frutas  

---

## 6. Compacidad (1 valor)

### ¿Qué es?

La **compacidad** mide **qué tan concentrada/densa es una forma**.

### Fórmula:

```
compacidad = (perímetro²) / área
```

### Rango de valores:

- **Compacidad baja** → Forma muy concentrada/compacta (redonda)
- **Compacidad alta** → Forma muy dispersa/alargada (delgada)

### Cálculo en el código:

```python
compactness = per*per/max_area               # Fórmula de compacidad
compactness_normalized = np.log10(np.abs(compactness)+d0)  # Normaliza con log10
```

### Ejemplos con tus frutas:

```
Naranja:
  - Forma redonda y compacta
  - compacidad ≈ 10-15 (BAJA/COMPACTA)

Banana:
  - Forma alargada y dispersa
  - compacidad ≈ 30-50 (ALTA/DISPERSA)

Manzana:
  - Forma semi-compacta
  - compacidad ≈ 15-25 (MEDIA)

Zapallo:
  - Forma dispersa
  - compacidad ≈ 20-40 (MEDIA-ALTA)
```

### Relación inversa con Circularidad:

| Métrica | Banana | Naranja |
|---------|--------|---------|
| **Circularidad** | 0.3 (baja) | 0.95 (alta) |
| **Compacidad** | 40 (alta) | 12 (baja) |

Son **opuestas**: cuando una sube, la otra baja.

### ¿Por qué ambas si son opuestas?

✅ **Redundancia controlada**: ofrecen información desde diferentes perspectivas  
✅ **Mayor discriminación**: el modelo ML tiene dos variables independientes  
✅ **Robustez**: si una falla, la otra aún funciona  

### Resumen:

- **Circularidad**: "¿qué tan redonda?" (0-1, mejor cuanto más cercano a 1)
- **Compacidad**: "¿qué tan dispersa?" (inversamente proporcional a circularidad)

Juntas forman un descriptor de forma muy efectivo para distinguir frutas.

---

## 7. Normalización Final

Después de concatenar todos los features:

```python
features = np.concatenate([
    hu_moments_normalized,      # 2 valores
    color_histograms_hsv,       # 512 valores (256 Hue + 256 Saturación)
    [circ_normalized, compactness_normalized]  # 2 valores
])

# Normalización final
features = cv2.normalize(features, features).flatten()
```

Se aplica una **normalización global** usando `cv2.normalize()` para que todos los valores estén en el rango [0, 1].

---

## 8. Resumen del Vector de Features

```
Vector final = [
  hu_1,                           # 1 valor (momento de Hu)
  hu_3,                           # 1 valor (momento de Hu)
  hist_hue[0..255],              # 256 valores (histograma Hue)
  hist_saturation[0..255],       # 256 valores (histograma Saturación)
  circ_normalized,               # 1 valor (circularidad normalizada)
  compactness_normalized         # 1 valor (compacidad normalizada)
]

TOTAL: 516 valores por imagen
```

---

## 9. Flujo Completo de Extracción de Features

```
Imagen Original
    ↓
Convertir a Escala de Grises
    ↓
Crear Máscara (umbral para eliminar fondo blanco)
    ↓
Invertir Máscara
    ↓
Aplicar Máscara a Imagen Original
    ↓
Redimensionar (200x200)
    ↓
Aplicar Filtro Bilateral (desenfoque)
    ↓
Aplicar Filtro de Sharpening
    ↓
Detección de Bordes (Canny)
    ↓
Encontrar Contornos
    ↓
Suavizar Contornos (Filtro Gaussiano)
    ↓
EXTRAER FEATURES:
    ├─ Hu Moments (2 valores)
    ├─ Histograma Hue (256 valores)
    ├─ Histograma Saturación (256 valores)
    ├─ Circularidad (1 valor)
    └─ Compacidad (1 valor)
    ↓
Normalizar Todos los Features
    ↓
Vector Final (516 valores)
```

---

## 10. ¿Cómo se Usan Estos Features?

Los 516 valores de cada imagen se envían al modelo **K-Means** que:

1. **Agrupa** imágenes similares (frutas del mismo tipo)
2. **Encuentra patrones**: qué combinación de features identifica mejor cada fruta
3. **Clasifica nuevas imágenes**: compara sus 516 features contra los clusters entrenados

---

## Conclusión

Este sistema de 516 features combina información de:
- ✅ **Forma**: Hu Moments, Circularidad, Compacidad
- ✅ **Color**: Histograma Hue, Histograma Saturación

Esto permite distinguir efectivamente entre banana, naranja, manzana y zapallo, incluso con variaciones en tamaño, orientación y estado de madurez.
