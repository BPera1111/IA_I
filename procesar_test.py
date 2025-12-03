"""
Script para procesar imágenes de test:
- Remover fondo con REMBG
- Guardar en data_sin_fondo/test_sin_fondo/
"""

import os
from PIL import Image
from rembg import remove


def process_test_images():
    """
    Procesa las imágenes de data/test/ y les remueve el fondo usando REMBG.
    Guarda los resultados en data_sin_fondo/test_sin_fondo/
    """
    
    # Directorios
    input_dir = "data/test"
    output_dir = "data_sin_fondo/test_sin_fondo"
    
    
    # Verificar que existe el directorio de entrada
    if not os.path.exists(input_dir):
        print(f"⚠ Error: No existe la carpeta {input_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar todas las imágenes en test/
    images = [f for f in os.listdir(input_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"⚠ No se encontraron imágenes en {input_dir}")
        return
    
    print("="*60)
    print(f"PROCESANDO IMÁGENES DE TEST CON REMBG")
    print("="*60)
    print(f"\nEncontradas {len(images)} imágenes en {input_dir}\n")
    
    processed = 0
    for img_name in images:
        try:
            # Ruta completa de entrada
            img_path = os.path.join(input_dir, img_name)
            
            # Abrir imagen y remover fondo
            print(f"Procesando: {img_name}...", end=" ")
            input_img = Image.open(img_path)
            output_img = remove(input_img)
            
            # Cambiar extensión a .png (para soportar transparencia)
            img_name_without_ext = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_dir, f'{img_name_without_ext}.png')
            
            # Guardar como PNG
            output_img.save(output_path, 'PNG')
            
            processed += 1
            print(f"✓ Guardado en {output_path}")
            
        except Exception as e:
            print(f"⚠ Error procesando {img_name}: {e}")
    
    print("\n" + "="*60)
    print(f"✓ COMPLETADO: {processed}/{len(images)} imágenes procesadas")
    print(f"✓ Imágenes guardadas en: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    process_test_images()
