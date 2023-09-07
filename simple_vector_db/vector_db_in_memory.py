from collections import defaultdict
from typing import List, Tuple

import numpy as np

from simple_vector_db.distances import cosine_similarity
from simple_vector_db.vector_db import VectorDB


class VectorDBInMemory(VectorDB):
    def __init__(self):
        self.vectors = defaultdict(np.ndarray)

    def insert(self, vectors_to_insert: list[np.ndarray]) -> None:
        for i in range(len(vectors_to_insert)):
            self.vectors[i] = vectors_to_insert[i]

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        similarities = [(key, cosine_similarity(query_vector, vector)) for key, vector in self.vectors.items()]
        # similarities = [(key, euclidean_distance(query_vector, vector)) for key, vector in self.vectors.items()]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def retrieve(self, key: int) -> np.ndarray:
        return self.vectors.get(key, None)
