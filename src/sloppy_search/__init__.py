import numpy as np

class Vector:
    def __init__(self, data):
        self.data = np.array(data)

class Database:
    def __init__(self):
        self.vectors = []

    def add_vector(self, vector):
        self.vectors.append(vector)
        print(f'Vector added: {vector.data}')  # Print statement added

    def search(self, query_vector, k=10):
        distances = np.linalg.norm(np.array([v.data for v in self.vectors]) - query_vector.data, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        print(f'Top {k} indices: {top_k_indices}')
        return distances[top_k_indices], top_k_indices
