import numpy as np
import pickle

class Database:
    """
    A class representing a database of vectors for similarity search.

    Attributes:
        vectors (list): A list of vectors in the database.

    Methods:
        add_vector: Add a vector to the database.
        search: Perform a similarity search on the database.
    """

    def __init__(self, filename=None):
        """
        Initialize the database.

        If a filename is provided, attempt to load the serialized object from the file.
        If no filename is provided, or if loading fails, initialize an empty database.

        Parameters:
        - filename (str, optional): The name of the file to load the serialized object. Defaults to None.

        Returns:
        None
        """
        if filename is not None:
            try:
                with open(filename + ".pkl", "rb") as file:
                    data = pickle.load(file)
                    for attr in vars(data):
                        setattr(self, attr, getattr(data, attr))
            except:
                print("Failed to load the serialized object from the file.")
                self.vectors = []
        else:
            self.vectors = []


    def add_vectors(self, vector):
        """
        Add a vector or a list of vectors to the existing vectors.

        Args:
            vector: A single vector or a list of vectors to be added.

        Returns:
            None
        """
        self.vectors.extend(vector)


    def search(self, query_vector, k=10):
        """
        Perform a similarity search on the database.

        Args:
            query_vector (np.array): The query vector for similarity search.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            distances (ndarray): An array of distances between the query vector and the nearest neighbors.
            top_k_indices (ndarray): An array of indices of the nearest neighbors in the database.
        """
        distances = np.linalg.norm(np.array(self.vectors) - query_vector, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        return distances[top_k_indices], top_k_indices
    

    def serialize(self, filename="database"):
        """
        Serialize the object using pickle.

        Parameters:
        - filename (str): The name of the file to save the serialized object. Defaults to "database".

        Returns:
        None
        """
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(self, file)
