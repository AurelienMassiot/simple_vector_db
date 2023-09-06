import sqlite3
from collections import defaultdict
from typing import List, Tuple


import numpy as np

from simple_vector_db.distances import cosine_similarity


class VectorDBInMemory:
    def __init__(self):
        self.vectors = defaultdict(np.ndarray)

    def insert(self, key: str, vector: np.ndarray) -> None:
        self.vectors[key] = vector

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        similarities = [(key, cosine_similarity(query_vector, vector)) for key, vector in self.vectors.items()]
        # similarities = [(key, euclidean_distance(query_vector, vector)) for key, vector in self.vectors.items()]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def retrieve(self, key: str) -> np.ndarray:
        return self.vectors.get(key, None)


class VectorDBSQLite:
    def __init__(self):
        self.conn = sqlite3.connect('vector_db.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, data BLOB)')

    def insert_many(self, vectors_to_insert: list[np.ndarray]) -> None:
        data_bytes = [(array.tobytes(),) for array in vectors_to_insert]
        self.cursor.executemany('INSERT INTO vectors (data) VALUES (?)', data_bytes)
        self.conn.commit()

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        self.cursor.execute('SELECT id, data FROM vectors')
        rows = self.cursor.fetchall()
        index_and_vectors = [(row[0], np.frombuffer(row[1], dtype=np.int64)) for row in rows]
        similarities = [(vector[0], cosine_similarity(query_vector, vector[1])) for vector in index_and_vectors]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def retrieve(self, key: int) -> np.ndarray:
        self.cursor.execute('SELECT id, data FROM vectors WHERE id = ?', (key,))
        row = self.cursor.fetchone()

        if row is not None:
            retrieved_index, retrieved_data = row[0], np.frombuffer(row[1], dtype=np.int64)
            return retrieved_index, retrieved_data
        else:
            return None

    def __del__(self):
        self.conn.close()
