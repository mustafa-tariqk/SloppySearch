import numpy as np

class Vector:
    def __init__(self, data):
        self.data = np.array(data)

class Database:
    def __init__(self):
        self.vectors = []

    def add_vector(self, vector):
        self.vectors.append(vector)

    def search(self, query_vector, k=10):
        distances = np.linalg.norm(np.array(self.vectors) - query_vector.data, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        return distances[top_k_indices], top_k_indices