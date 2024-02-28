import numpy as np

class Vector:
    def __init__(self, data):
        self.data = np.array(data)

class Database:
    """
    A class representing a database of vectors for similarity search.

    Attributes:
        vectors (list): A list of vectors in the database.

    Methods:
        add_vector: Add a vector to the database.
        search: Perform a similarity search on the database.
    """

    def __init__(self):
        self.vectors = []

    def add_vector(self, vector):
        """
        Add a vector to the database.

        Args:
            vector: The vector to be added.
        """
        self.vectors.append(vector)

    def search(self, query_vector, k=10):
        """
        Perform a similarity search on the database.

        Args:
            query_vector: The query vector for similarity search.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            distances (ndarray): An array of distances between the query vector and the nearest neighbors.
            top_k_indices (ndarray): An array of indices of the nearest neighbors in the database.
        """
        distances = np.linalg.norm(np.array([v.data for v in self.vectors]) - query_vector.data, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        return distances[top_k_indices], top_k_indices
