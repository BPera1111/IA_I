import numpy as np


class SimpleKMeans:
    def __init__(self, k, max_iters=100, tol=1e-4, random_state=None):
        """
        k: cantidad de clusters
        max_iters: máximo de iteraciones del algoritmo
        tol: tolerancia para detectar convergencia (cambio de centroides)
        random_state: semilla opcional para hacer reproducible la inicialización
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids = None  # se va a llenar en fit()
    
    def _init_centroids(self, X):
        """
        Elige k puntos al azar de X como centroides iniciales.
        X: array de shape (n_muestras, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]
        if self.k > n_samples:
            raise ValueError("k no puede ser mayor que la cantidad de muestras")

        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices].copy()

    def _compute_distances(self, X):
        """
        Calcula la distancia euclidiana de cada punto a cada centroide.
        Devuelve un array de shape (n_muestras, k)
        """
        # X: (n_muestras, n_features)
        # centroids: (k, n_features)
        # Usamos broadcasting de numpy: (n_muestras, 1, n_features) - (1, k, n_features)
        diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)  # (n_muestras, k)
        distances = np.sqrt(dist_sq)
        return distances

    def fit(self, X):
        """
        Ejecuta el algoritmo K-Means sobre los datos X.
        X: array de shape (n_muestras, n_features)
        """
        X = np.asarray(X, dtype=np.float32)
        self._init_centroids(X)

        for _ in range(self.max_iters):
            # 1) Asignar cada punto al centroide más cercano
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)  # (n_muestras,)

            # 2) Recalcular centroides
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.k):
                cluster_points = X[labels == j]
                if len(cluster_points) == 0:
                    # Si un cluster queda vacío, reasignamos su centroide
                    # a un punto aleatorio
                    idx = np.random.choice(X.shape[0])
                    new_centroids[j] = X[idx]
                else:
                    new_centroids[j] = np.mean(cluster_points, axis=0)

            # 3) Revisar convergencia (si los centroides cambiaron muy poco)
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

    def predict(self, X):
        """
        Asigna cada punto en X al cluster más cercano.
        X: array de shape (n_muestras, n_features)
        Devuelve: array de shape (n_muestras,) con labels en [0, k-1]
        """
        if self.centroids is None:
            raise ValueError("Debes llamar a fit(X) antes de predict().")

        X = np.asarray(X, dtype=np.float32)
        distances = self._compute_distances(X)
        labels = np.argmin(distances, axis=1)
        return labels

    def predict_one(self, x):
        """
        Asigna un solo vector x al cluster más cercano.
        x: array de shape (n_features,)
        Devuelve: entero en [0, k-1]
        """
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        labels = self.predict(x)
        return int(labels[0])
