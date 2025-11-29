# archivo: test_kmeans.py

from image import FruitDataset
from kmeans import SimpleKMeans

if __name__ == "__main__":
    root = "/home/bruno/fing/IA_I/img/"

    ds = FruitDataset(root)
    ds.build(sat_threshold=40, central_fraction=0.6)
    X, y = ds.get_data()

    print("Shape X:", X.shape)
    print("Etiquetas:", y)

    # Probamos K-Means con k=4 (manzana, banana, pera, naranja)
    kmeans = SimpleKMeans(k=4, max_iters=100, tol=1e-4, random_state=42)
    kmeans.fit(X)

    print("Centroides encontrados (H, S, V):")
    print(kmeans.centroids)

    # Cluster asignado a las primeras 8 muestras
    labels = kmeans.predict(X)
    for i in range(min(8, len(X))):
        print(f"i={i}, vec={X[i]}, etiqueta_real={y[i]}, cluster={labels[i]}")
