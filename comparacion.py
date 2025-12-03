import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rembg import remove
import time

print("Todas las librerías importadas correctamente")

# Buscar una imagen en la carpeta data
data_path = "data"
fruit_folders = ['banana', 'manzana', 'naranja', 'zapallo']

# Obtener primera imagen disponible
test_image_path = None
for fruit in fruit_folders:
    fruit_path = os.path.join(data_path, fruit)
    if os.path.exists(fruit_path):
        images = [f for f in os.listdir(fruit_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            test_image_path = os.path.join(fruit_path, images[0])
            break

if test_image_path:
    print(f"Imagen de prueba: {test_image_path}")
    original_img = Image.open(test_image_path)
    print(f"Tamaño: {original_img.size}")
else:
    print("No se encontraron imágenes en data/")



def remove_bg_rembg(image_path):
    """Remover fondo usando rembg"""
    start_time = time.time()
    
    input_img = Image.open(image_path)
    output_img = remove(input_img)
    
    elapsed_time = time.time() - start_time
    return output_img, elapsed_time

# Probar método 1
print("Ejecutando rembg...")
result_rembg, time_rembg = remove_bg_rembg(test_image_path)
print(f"✓ Tiempo: {time_rembg:.2f}s")


# def remove_bg_grabcut(image_path):
#     """Remover fondo usando GrabCut de OpenCV"""
#     start_time = time.time()
    
#     # Leer imagen
#     img = cv2.imread(image_path)
    
#     # Inicializar máscara
#     mask = np.zeros(img.shape[:2], np.uint8)
    
#     # Definir rectángulo aproximado del objeto (dejando margen)
#     h, w = img.shape[:2]
#     rect = (10, 10, w-20, h-20)
    
#     # Aplicar GrabCut
#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)
#     cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
#     # Crear máscara final
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#     output = img * mask2[:, :, np.newaxis]
    
#     # Convertir a PIL
#     output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
#     output_pil = Image.fromarray(output_rgb)
    
#     elapsed_time = time.time() - start_time
#     return output_pil, elapsed_time

# Probar método 2
# print("Ejecutando GrabCut...")
# result_grabcut, time_grabcut = remove_bg_grabcut(test_image_path)
# print(f"✓ Tiempo: {time_grabcut:.2f}s")


# def remove_bg_maskrcnn(image_path):
#     """Remover fondo usando Mask R-CNN"""
#     start_time = time.time()
    
#     try:
#         import torchvision
#         from torchvision import transforms
#         import torch
#     except ImportError:
#         print("Instalando dependencias de Mask R-CNN...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "-q"])
#         import torch
#         from torchvision import transforms
    
#     # Cargar modelo preentrenado
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#     model.eval()
    
#     # Preparar imagen
#     img = Image.open(image_path).convert('RGB')
#     img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    
#     # Predicción
#     with torch.no_grad():
#         predictions = model(img_tensor)
    
#     # Procesar máscara
#     masks = predictions[0]['masks']
#     scores = predictions[0]['scores']
    
#     if len(masks) > 0 and scores[0] > 0.5:
#         # Usar la máscara con mayor puntuación
#         mask = masks[0, 0].cpu().numpy()
#         mask = (mask > 0.5).astype(np.uint8) * 255
        
#         # Aplicar máscara a imagen
#         img_array = np.array(img)
#         # Redimensionar máscara si es necesario
#         mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
        
#         img_array[mask_resized == 0] = [255, 255, 255]
#         output_pil = Image.fromarray(img_array.astype('uint8'))
#     else:
#         output_pil = img
    
#     elapsed_time = time.time() - start_time
#     return output_pil, elapsed_time

# # Probar método 3 (puede tardar más)
# print("Ejecutando Mask R-CNN (esto puede tardar más)...")
# try:
#     result_maskrcnn, time_maskrcnn = remove_bg_maskrcnn(test_image_path)
#     print(f"✓ Tiempo: {time_maskrcnn:.2f}s")
# except Exception as e:
#     print(f"⚠ Error con Mask R-CNN: {e}")
#     result_maskrcnn = None
#     time_maskrcnn = None


#     # Crear figura comparativa
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Fila 1: Resultados de remoción de fondo
# Original
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(original_img)
ax1.set_title('Imagen Original', fontsize=12, fontweight='bold')
ax1.axis('off')

# REMBG
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(result_rembg)
ax2.set_title(f'REMBG\n({time_rembg:.2f}s)', fontsize=12, fontweight='bold', color='green')
ax2.axis('off')

# # GrabCut
# ax3 = fig.add_subplot(gs[0, 2])
# ax3.imshow(result_grabcut)
# ax3.set_title(f'GrabCut\n({time_grabcut:.2f}s)', fontsize=12, fontweight='bold', color='blue')
# ax3.axis('off')

# # Fila 2: Detalles y comparación
# if result_maskrcnn:
#     # Mask R-CNN
#     ax4 = fig.add_subplot(gs[1, 0])
#     ax4.imshow(result_maskrcnn)
#     ax4.set_title(f'Mask R-CNN\n({time_maskrcnn:.2f}s)', fontsize=12, fontweight='bold', color='red')
#     ax4.axis('off')
# else:
#     ax4 = fig.add_subplot(gs[1, 0])
#     ax4.text(0.5, 0.5, 'Mask R-CNN\nno disponible', ha='center', va='center', fontsize=12)
#     ax4.axis('off')

# # Tabla comparativa
# ax5 = fig.add_subplot(gs[1, 1:])
# ax5.axis('off')

# comparison_data = [
#     ['Método', 'Tiempo (s)', 'Calidad', 'Facilidad', 'Recursos'],
#     ['REMBG', f'{time_rembg:.2f}', 'Muy Buena', 'Muy Fácil', 'Bajo'],
#     ['GrabCut', f'{time_grabcut:.2f}', 'Buena', 'Media', 'Bajo'],
#     ['Mask R-CNN', f'{time_maskrcnn:.2f}' if time_maskrcnn else 'N/A', 'Excelente', 'Difícil', 'Alto']
# ]

# table = ax5.table(cellText=comparison_data, cellLoc='center', loc='center',
#                    colWidths=[0.2, 0.15, 0.2, 0.2, 0.15])
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1, 2)

# # Colorear encabezado
# for i in range(5):
#     table[(0, i)].set_facecolor('#40466e')
#     table[(0, i)].set_text_props(weight='bold', color='white')

# plt.suptitle('Comparación de 3 Métodos para Remover Fondo', fontsize=16, fontweight='bold', y=0.98)
# plt.tight_layout()
plt.show()

# print("\n✓ Comparación completada")