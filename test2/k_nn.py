import numpy as np
import collections

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.data = None
        self.labels = None
        self.n_examples = 0
    
    def learning(self, data, labels):
        self.data = data
        self.labels = labels
        self.n_examples = self.data.shape[0]
    
    def predict(self, test):
        classes = []
        distances = np.empty(self.n_examples)
        for i in range(self.n_examples):
            distances[i] = self.euclidean_distance(self.data[i], test)

        k_dist = np.argsort(distances) #K distancias mas cercanas
        k_labels = self.labels[k_dist[:self.k]] #K etiquetas de las distancias mas cercanas
        count = collections.Counter(k_labels).most_common(1) #Conteo de las etiquetas
        classes.append(count[0][0])
        
        return classes[0]

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))