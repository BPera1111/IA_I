"""
Script simple para entrenar y visualizar K-Means en el dataset de frutas.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from image import FruitImage, FruitDataset
from kmeans import SimpleKMeans


def main():
    # ============================================================
    # 1. CARGAR DATASET
    # ============================================================
    print("="*60)
    print("1. CARGANDO DATASET")
    print("="*60)
    
    root_dir = "/home/bruno/fing/IA_I/data_sin_fondo"
    ds = FruitDataset(root_dir)
    ds.build(sat_threshold=40, central_fraction=0.6)
    
    X, y = ds.get_data()
    
    # Ahora X tiene 8 features: [cos(H), sin(H), S, V, hu0, hu1, circ, compact]
    print(f"Shape X: {X.shape}")
    print(f"Cantidad de muestras: {len(X)}")
    print(f"Clases presentes: {np.unique(y)}\n")
    
    
    # ============================================================
    # 2. MOSTRAR ESTADÍSTICAS DEL DATASET
    # ============================================================
    print("="*60)
    print("2. ESTADÍSTICAS DEL DATASET")
    print("="*60)
    
    print(f"\nVectores [cos(H), sin(H), S, V, Hu0, Hu1, Circ, Compact] de todas las imágenes:\n")
    print(f"{'Idx':<5} {'cos(H)':<8} {'sin(H)':<8} {'S':<6} {'V':<6} {'Hu0':<6} {'Hu1':<6} {'Circ':<6} {'Comp':<6} {'Clase':<20}")
    print("-"*90)
    
    for i in range(len(X)):
        cos_h, sin_h, s, v, hu0, hu1, circ, compact = X[i]
        clase = y[i]
        print(f"{i:<5} {cos_h:<8.3f} {sin_h:<8.3f} {s:<6.2f} {v:<6.2f} {hu0:<6.2f} {hu1:<6.2f} {circ:<6.2f} {compact:<6.2f} {clase:<20}")
    
    print("\n" + "="*60)
    print("Estadísticas por canal (8 features):")
    print(f"cos(H)   - Min: {X[:, 0].min():.3f}, Max: {X[:, 0].max():.3f}, Media: {X[:, 0].mean():.3f}")
    print(f"sin(H)   - Min: {X[:, 1].min():.3f}, Max: {X[:, 1].max():.3f}, Media: {X[:, 1].mean():.3f}")
    print(f"Sat      - Min: {X[:, 2].min():.3f}, Max: {X[:, 2].max():.3f}, Media: {X[:, 2].mean():.3f}")
    print(f"Val      - Min: {X[:, 3].min():.3f}, Max: {X[:, 3].max():.3f}, Media: {X[:, 3].mean():.3f}")
    print(f"Hu0      - Min: {X[:, 4].min():.3f}, Max: {X[:, 4].max():.3f}, Media: {X[:, 4].mean():.3f}")
    print(f"Hu1      - Min: {X[:, 5].min():.3f}, Max: {X[:, 5].max():.3f}, Media: {X[:, 5].mean():.3f}")
    print(f"Circ     - Min: {X[:, 6].min():.3f}, Max: {X[:, 6].max():.3f}, Media: {X[:, 6].mean():.3f}")
    print(f"Compact  - Min: {X[:, 7].min():.3f}, Max: {X[:, 7].max():.3f}, Media: {X[:, 7].mean():.3f}")
    
    print("\nEstadísticas por clase:")
    for clase in np.unique(y):
        idx = np.where(y == clase)[0]
        print(f"\n{clase.upper()}:")
        print(f"  Cantidad: {len(idx)}")
        print(f"  cos(H)  - Media: {X[idx, 0].mean():.3f}, Std: {X[idx, 0].std():.3f}")
        print(f"  sin(H)  - Media: {X[idx, 1].mean():.3f}, Std: {X[idx, 1].std():.3f}")
    # ============================================================
    # 3. ENTRENAR K-MEANS con 8 features
    # ============================================================
    print("="*60)
    print("3. ENTRENANDO K-MEANS [cos(H), sin(H), S, V, Hu0, Hu1, Circ, Compact]")
    print("="*60)
    
    k = 4
    kmeans = SimpleKMeans(k=k, max_iters=100, tol=1e-4, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    
    print(f"\nCentroides encontrados (8 features):")
    for i, c in enumerate(kmeans.centroids):
        # Reconstruir Hue para visualizar
        hue_angle = np.arctan2(c[1], c[0]) * (180.0 / np.pi)
        if hue_angle < 0:
            hue_angle += 360
        print(f"  Cluster {i}:")
        print(f"    Color: Hue≈{hue_angle:.0f}°, S={c[2]:.2f}, V={c[3]:.2f}")
        print(f"    Forma: Hu0={c[4]:.2f}, Hu1={c[5]:.2f}, Circ={c[6]:.2f}, Compact={c[7]:.2f}")
    
    print(f"\nCentroides encontrados [cos(H), sin(H), S, V]:")
    for i, c in enumerate(kmeans.centroids):
        # Reconstruir Hue para visualizar
        hue_angle = np.arctan2(c[1], c[0]) * (180.0 / np.pi)
        if hue_angle < 0:
            hue_angle += 360
        print(f"  Cluster {i}: cos={c[0]:.3f}, sin={c[1]:.3f}, S={c[2]:.1f}, V={c[3]:.1f} [Hue≈{hue_angle:.1f}°]")
    
    print(f"\nDistribución de clusters:")
    for cluster_id in range(k):
        count = np.sum(labels == cluster_id)
        cluster_indices = np.where(labels == cluster_id)[0]
        clases_en_cluster = [y[i] for i in cluster_indices]
        print(f"  Cluster {cluster_id}: {count} imágenes - {clases_en_cluster}")
    
    
    # ============================================================
    # 4. MOSTRAR IMÁGENES ORGANIZADAS POR CLUSTER
    # ============================================================
    print("\n" + "="*60)
    print("4. MOSTRANDO IMÁGENES POR CLUSTER...")
    print("="*60)
    
    # Recolectar rutas de imágenes
    image_paths = []
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(label_path, filename)
                image_paths.append(img_path)
    
    # Crear grid para mostrar imágenes por cluster
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        n_images = len(cluster_indices)
        
        if n_images == 0:
            continue
        
        # Calcular dimensiones del grid
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        fig.suptitle(f"Cluster {cluster_id} ({n_images} imágenes)", fontsize=14, fontweight='bold')
        
        for plot_idx, img_idx in enumerate(cluster_indices):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            img_path = image_paths[img_idx]
            # Leer con IMREAD_UNCHANGED para PNG con transparencia
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                # Convertir a RGB según canales
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # BGRA (PNG con transparencia)
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    else:  # BGR
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:  # Escala de grises
                    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                ax.imshow(rgb)
                
                clase_real = y[img_idx]
                cos_h, sin_h, s, v, hu0, hu1, circ, compact = X[img_idx]
                # Reconstruir Hue para mostrar
                hue_angle = np.arctan2(sin_h, cos_h) * (180.0 / np.pi)
                if hue_angle < 0:
                    hue_angle += 360
                # Mostrar features principales
                ax.set_title(f"{clase_real}\nH:{hue_angle:.0f}° S:{s:.2f} V:{v:.2f}\nCirc:{circ:.2f} Comp:{compact:.2f}", fontsize=8)
            
            ax.axis('off')
        
        # Ocultar ejes vacíos
        for plot_idx in range(n_images, n_rows * n_cols):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'/home/bruno/fing/IA_I/test/cluster_{cluster_id}_imagenes.png', dpi=100, bbox_inches='tight')
        print(f"✓ Cluster {cluster_id}: {n_images} imágenes guardadas")
    
    print(f"\n✓ Visualización de imágenes por cluster completada")
    plt.show()
    
    
    print("\n" + "="*60)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
