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
    
    root_dir = "/home/bruno/fing/IA_I/img/"
    ds = FruitDataset(root_dir)
    ds.build(sat_threshold=40, central_fraction=0.6)
    
    X, y = ds.get_data()
    
    print(f"Shape X: {X.shape}")
    print(f"Cantidad de muestras: {len(X)}")
    print(f"Clases presentes: {np.unique(y)}\n")
    
    
    # ============================================================
    # 2. MOSTRAR ESTADÍSTICAS DEL DATASET
    # ============================================================
    print("="*60)
    print("2. ESTADÍSTICAS DEL DATASET")
    print("="*60)
    
    print(f"\nVectores HSV de todas las imágenes:\n")
    print(f"{'Idx':<5} {'Hue':<8} {'Sat':<8} {'Val':<8} {'Clase':<12}")
    print("-"*50)
    
    for i in range(len(X)):
        h, s, v = X[i]
        clase = y[i]
        print(f"{i:<5} {h:<8.1f} {s:<8.1f} {v:<8.1f} {clase:<12}")
    
    print("\n" + "="*60)
    print("Estadísticas por canal HSV:")
    print(f"Hue   - Min: {X[:, 0].min():.1f}, Max: {X[:, 0].max():.1f}, Media: {X[:, 0].mean():.1f}")
    print(f"Sat   - Min: {X[:, 1].min():.1f}, Max: {X[:, 1].max():.1f}, Media: {X[:, 1].mean():.1f}")
    print(f"Val   - Min: {X[:, 2].min():.1f}, Max: {X[:, 2].max():.1f}, Media: {X[:, 2].mean():.1f}")
    
    print("\nEstadísticas por clase:")
    for clase in np.unique(y):
        idx = np.where(y == clase)[0]
        print(f"\n{clase.upper()}:")
        print(f"  Cantidad: {len(idx)}")
        print(f"  Hue   - Media: {X[idx, 0].mean():.1f}, Std: {X[idx, 0].std():.1f}")
        print(f"  Sat   - Media: {X[idx, 1].mean():.1f}, Std: {X[idx, 1].std():.1f}")
        print(f"  Val   - Media: {X[idx, 2].mean():.1f}, Std: {X[idx, 2].std():.1f}\n")
    
    
    # ============================================================
    # 3. ENTRENAR K-MEANS (H, S, V)
    # ============================================================
    print("="*60)
    print("3. ENTRENANDO K-MEANS (H, S, V)")
    print("="*60)
    
    k = 4
    kmeans = SimpleKMeans(k=k, max_iters=100, tol=1e-4, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    
    print(f"\nCentroides encontrados (H, S, V):")
    for i, c in enumerate(kmeans.centroids):
        print(f"  Cluster {i}: {c}")
    
    print(f"\nAsignaciones (primeras 10):")
    for i in range(min(10, len(X))):
        print(f"  i={i:02d} | vec={X[i]} | clase_real={y[i]} | cluster={labels[i]}")
    
    print(f"\nDistribución de clusters:")
    for cluster_id in range(k):
        count = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} imágenes")
    

    
    
    # ============================================================
    # 6. VISUALIZACIONES 3D
    # ============================================================
    print("\n" + "="*60)
    print("6. GENERANDO VISUALIZACIONES 3D...")
    print("="*60)
    
    unique_classes = np.unique(y)
    colors_map = {'manzana': 'red', 'banana': 'yellow', 'pera': 'green', 'naranja': 'orange'}
    
    # Gráfico 1: HSV 3D - Por Clusters
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="tab10", s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:, 2], 
                c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroides')
    ax1.set_xlabel("Hue (H)")
    ax1.set_ylabel("Saturación (S)")
    ax1.set_zlabel("Valor (V)")
    ax1.set_title("Espacio HSV 3D - Coloreado por Cluster")
    plt.colorbar(scatter1, ax=ax1, label="Cluster", shrink=0.8)
    ax1.legend()
    
    # Gráfico 2: HSV 3D - Por Clases Reales
    ax2 = fig.add_subplot(122, projection='3d')
    for clase in unique_classes:
        idx = np.where(y == clase)[0]
        ax2.scatter(X[idx, 0], X[idx, 1], X[idx, 2], label=clase, s=50, alpha=0.7, 
                   color=colors_map.get(clase, 'gray'), edgecolors='black', linewidth=0.5)
    ax2.set_xlabel("Hue (H)")
    ax2.set_ylabel("Saturación (S)")
    ax2.set_zlabel("Valor (V)")
    ax2.set_title("Espacio HSV 3D - Coloreado por Clase Real")
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/bruno/fing/IA_I/test/resultado_kmeans_3d.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gráfico guardado en: resultado_kmeans_3d.png")
    plt.show()
    
    
    # ============================================================
    # 7. MOSTRAR IMÁGENES ORGANIZADAS POR CLUSTER
    # ============================================================
    print("\n" + "="*60)
    print("7. MOSTRANDO IMÁGENES POR CLUSTER...")
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
            bgr = cv2.imread(img_path)
            
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
                clase_real = y[img_idx]
                h, s, v = X[img_idx]
                ax.set_title(f"{clase_real}\nH:{h:.0f} S:{s:.0f} V:{v:.0f}", fontsize=9)
            
            ax.axis('off')
        
        # Ocultar ejes vacíos
        for plot_idx in range(n_images, n_rows * n_cols):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'/home/bruno/fing/IA_I/test/cluster_{cluster_id}_imagenes.png', dpi=100, bbox_inches='tight')
        print(f"✓ Cluster {cluster_id}: {n_images} imágenes")
    
    print(f"\n✓ Visualización de imágenes por cluster completada")
    plt.show()
    
    
    print("\n" + "="*60)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*60)
    print("\nPróximos pasos:")
    print("1. Revisar los gráficos para entender dónde está el problema")
    print("2. Si manzana/naranja se solapan en HSV, usa SV en lugar")
    print("3. Ajusta parámetros en image.py si es necesario")


if __name__ == "__main__":
    main()
