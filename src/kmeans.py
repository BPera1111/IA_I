import numpy as np


class SimpleKMeans:
    def __init__(self, k, max_iters=10000, tol=1e-4, random_state=None):
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
        Inicializa centroides usando K-Means++ para mejor distribución inicial.
        X: array de shape (n_muestras, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]
        if self.k > n_samples:
            raise ValueError("k no puede ser mayor que la cantidad de muestras")

        # K-Means++: inicialización inteligente
        self.centroids = np.zeros((self.k, X.shape[1]))
        
        # 1. Elegir primer centroide al azar
        first_idx = np.random.randint(n_samples)
        self.centroids[0] = X[first_idx]
        
        # 2. Para cada centroide restante
        for i in range(1, self.k):
            # Calcular distancia mínima de cada punto a los centroides ya elegidos
            distances = np.array([
                np.min([np.linalg.norm(x - c) for c in self.centroids[:i]])
                for x in X
            ])
            
            # Elegir siguiente centroide con probabilidad proporcional a distancia²
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            
            next_idx = np.random.choice(n_samples, p=probabilities)
            self.centroids[i] = X[next_idx]

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
