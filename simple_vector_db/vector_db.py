import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import KMeans

from simple_vector_db.distances import cosine_similarity


class VectorDB(ABC):
    @abstractmethod
    def insert(self, vectors_to_insert):
        pass

    @abstractmethod
    def search(self, query_vector, k):
        pass

    @abstractmethod
    def retrieve(self, key):
        pass


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


class VectorDBSQLite(VectorDB):
    def __init__(self):
        self.conn = sqlite3.connect('vector_db.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, data FLOAT)')

    def insert(self, vectors_to_insert: list[np.ndarray]) -> None:
        data_bytes = [(array.tobytes(),) for array in vectors_to_insert]
        self.cursor.executemany('INSERT INTO vectors (data) VALUES (?)', data_bytes)
        self.conn.commit()

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        self.cursor.execute('SELECT id, data FROM vectors')
        rows = self.cursor.fetchall()
        index_and_vectors = [(self._convert_row(row)) for row in rows]
        similarities = [(vector[0], cosine_similarity(query_vector, vector[1])) for vector in index_and_vectors]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def retrieve(self, key: int) -> Optional[tuple[Any, ndarray]]:
        self.cursor.execute('SELECT id, data FROM vectors WHERE id = ?', (key,))
        row = self.cursor.fetchone()

        if row is not None:
            retrieved_index, retrieved_data = self._convert_row(row)
            return retrieved_index, retrieved_data
        else:
            return None

    def index(self, n_clusters):
        self.cursor.execute('SELECT id, data FROM vectors')
        rows = self.cursor.fetchall()
        index_and_vectors = [(self._convert_row(row)) for row in rows]
        df_vectors = pd.DataFrame(index_and_vectors, columns=['Index', 'Vector'])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        df_vectors['Cluster'] = kmeans.fit_predict(list(df_vectors['Vector']))
        df_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['x', 'y', 'z'])

        return df_vectors, df_centroids

    def _convert_row(self, row) -> Tuple[int, ndarray]:
        return row[0], np.frombuffer(row[1], dtype=np.float64)

    def __del__(self):
        self.conn.close()
