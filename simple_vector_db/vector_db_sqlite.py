import sqlite3
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import KMeans

from simple_vector_db.distances import cosine_similarity
from simple_vector_db.vector_db import VectorDB


class VectorDBSQLite(VectorDB):
    def __init__(self, db_filename: str):
        self.conn = sqlite3.connect(db_filename)
        self.cursor = self.conn.cursor()

    def insert(self, vectors: list[np.ndarray]) -> None:
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, data FLOAT)"
        )
        vectors_bytes = [self._array_to_bytes(array) for array in vectors]
        self.cursor.executemany("INSERT INTO vectors (data) VALUES (?)", vectors_bytes)
        self.conn.commit()

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        self.cursor.execute("SELECT id, data FROM vectors")
        rows = self.cursor.fetchall()
        index_and_vectors = [(self._convert_row(row)) for row in rows]
        similarities = self.compute_similarities(index_and_vectors, query_vector)
        return similarities[:k]

    def compute_similarities(self, index_and_vectors, query_vector):
        similarities = [
            (vector[0], cosine_similarity(query_vector, vector[1]))
            for vector in index_and_vectors
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def retrieve(self, key: int) -> Optional[tuple[Any, ndarray]]:
        self.cursor.execute("SELECT id, data FROM vectors WHERE id = ?", (key,))
        row = self.cursor.fetchone()

        if row is not None:
            retrieved_index, retrieved_data = self._convert_row(row)
            return retrieved_index, retrieved_data
        else:
            return None

    def create_index_kmeans(self, n_clusters: int):
        self.cursor.execute("SELECT id, data FROM vectors")
        rows = self.cursor.fetchall()
        index_and_vectors = [(self._convert_row(row)) for row in rows]
        df_vectors = pd.DataFrame(index_and_vectors, columns=["Index", "Vector"])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        df_vectors["Cluster"] = kmeans.fit_predict(list(df_vectors["Vector"]))

        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS centroids_kmeans (id INTEGER PRIMARY KEY, data FLOAT)"
        )
        data_bytes = [(centroid.tobytes(),) for centroid in kmeans.cluster_centers_]
        self.cursor.executemany(
            "INSERT INTO centroids_kmeans (data) VALUES (?)", data_bytes
        )
        self.conn.commit()

        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS indexed_vectors_kmeans (id INTEGER PRIMARY KEY, data FLOAT, cluster INTEGER)"
        )
        index_and_vectors_and_clusters = [
            (t[0], t[1].tobytes(), int(v + 1))
            for t, v in zip(index_and_vectors, kmeans.labels_)
        ]  # v+1 pour avoir des clusters de 1 à n_clusters et pas de 0 à n_clusters-1, car l'insertion sur la primary key va aller de 1 à N-clusters
        self.cursor.executemany(
            "INSERT INTO indexed_vectors_kmeans (id, data, cluster) VALUES (?, ?, ?)",
            index_and_vectors_and_clusters,
        )
        self.conn.commit()

        return df_vectors

    def search_in_kmeans_index(
            self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[str, float]]:
        self.cursor.execute("SELECT id, data FROM centroids_kmeans")
        rows = self.cursor.fetchall()
        index_and_vectors = [(self._convert_row(row)) for row in rows]
        centroid_similarities = self.compute_similarities(index_and_vectors, query_vector)
        most_similar_centroid = centroid_similarities[0][0]

        self.cursor.execute(
            "SELECT id, data FROM indexed_vectors_kmeans WHERE cluster = ?",
            (most_similar_centroid,),
        )
        rows = self.cursor.fetchall()
        index_and_vectors = [(self._convert_row(row)) for row in rows]
        vectors_similarities = self.compute_similarities(index_and_vectors, query_vector)
        most_similar_vectors = vectors_similarities[:k]

        return most_similar_vectors, most_similar_centroid, centroid_similarities

    def _convert_row(self, row) -> Tuple[int, ndarray]:
        return row[0], self._bytes_to_array(row[1])

    def _array_to_bytes(self, array: np.ndarray) -> tuple[bytes]:
        return (array.tobytes(),)

    def _bytes_to_array(self, bytes: bytes) -> np.ndarray:
        return np.frombuffer(bytes, dtype=np.float64)

    def __del__(self):
        self.conn.close()
