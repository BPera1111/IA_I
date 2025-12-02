import numpy as np
from scipy.spatial import distance

class KMeans:
    def __init__(self, n_clusters, max_iter=50):  # Constructor que inicializa el número de clusters y el máximo de iteraciones
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def spatial_distance(self, X, Y):  # Calcula la distancia euclidiana entre dos puntos X e Y
        return distance.cdist(X, Y, 'euclidean')

    def fit(self, X, centroids=None):  # Entrena al modelo para encontrar los centroides y asignar puntos a los clusters
        if centroids is None:
            random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids = X[random_indices]
        else:
            self.centroids = centroids

        for _ in range(self.max_iter):  # Ajuste de KMeans
            distances = self.spatial_distance(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return labels

    def predict(self, X):  # Predice el cluster más cercano para nuevos puntos, usando los centroides ajustados
        distances = self.spatial_distance(X, self.centroids)
        return np.argmin(distances, axis=1)
