import numpy as np
from collections import defaultdict
from typing import List, Tuple
from simple_vector_db.distances import cosine_similarity, euclidean_distance



class VectorDatabase:
    def __init__(self):
        self.vectors = defaultdict(np.ndarray)

    def insert(self, key: str, vector: np.ndarray) -> None:
        self.vectors[key] = vector

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        similarities = [(key, cosine_similarity(query_vector, vector)) for key, vector in self.vectors.items()]
        #similarities = [(key, euclidean_distance(query_vector, vector)) for key, vector in self.vectors.items()]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def retrieve(self, key: str) -> np.ndarray:
        return self.vectors.get(key, None)