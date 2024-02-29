import numpy as np
import pickle

class Database:
    """
    A class representing a database for storing and searching vectors.

    Attributes:
        num_hash_tables (int): The number of hash tables to use for indexing.
        hash_size (int): The size of each hash table.
        hash_tables (list): A list of hash tables used for indexing vectors.

    Methods:
        add_vectors(vectors): Add vectors to the database.
        search(query_vector, k): Search for the nearest neighbors of a query vector.
        serialize(filename): Serialize the database to a file.
        deserialize(filename): Deserialize the database from a file.
    """

    def __init__(self, num_hash_tables=15, hash_size=200):
        self.num_hash_tables = num_hash_tables
        self.hash_size = hash_size
        self.hash_tables = [{} for _ in range(num_hash_tables)]

    def add_vectors(self, vectors):
        """
        Add vectors to the database.

        Args:
            vectors (list): A list of vectors to add to the database.
        """
        vectors = np.array(vectors)
        hashed_vectors = self.hash_vector(vectors)
        for i in range(self.num_hash_tables):
            hash_vals = hashed_vectors[:, i]
            unique_hash_vals, counts = np.unique(hash_vals, return_counts=True)
            for hash_val, count in zip(unique_hash_vals, counts):
                indices = np.where(hash_vals == hash_val)[0]
                if hash_val not in self.hash_tables[i]:
                    self.hash_tables[i][hash_val] = []
                self.hash_tables[i][hash_val].extend(vectors[indices])

    def search(self, query_vector, k=10):
        """
        Search for the nearest neighbors of a query vector.

        Args:
            query_vector (list): The query vector.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            list: The k nearest neighbors of the query vector.
        """
        hashed_query = self.hash_vector(query_vector)
        candidates = set()
        for i in range(self.num_hash_tables):
            hash_val = hashed_query[i]
            if hash_val in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][hash_val])
        candidates = np.array(list(candidates))
        distances = np.linalg.norm(candidates - np.array(query_vector), axis=1)
        sorted_indices = np.argsort(distances)
        return candidates[sorted_indices[:k]]

    def serialize(self, filename="database"):
        """
        Serialize the database to a file.

        Args:
            filename (str): The name of the file to save the serialized database to.
        """
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(self, file)

    def deserialize(self, filename):
        """
        Deserialize the database from a file.

        Args:
            filename (str): The name of the file to deserialize the database from.
        """
        with open(filename + ".pkl", "rb") as file:
            data = pickle.load(file)
            for attr in vars(data):
                setattr(self, attr, getattr(data, attr))
