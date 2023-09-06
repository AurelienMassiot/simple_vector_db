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

    def retrieve(self, key: str) -> np.ndarray:
        self.cursor.execute('SELECT id, data FROM vectors WHERE id = ?', (key,))
        row = self.cursor.fetchone()

        if row is not None:
            retrieved_index, retrieved_data = row[0], np.frombuffer(row[1], dtype=np.int64)
            return retrieved_index, retrieved_data
        else:
            return None

    def __del__(self):
        self.conn.close()

# def create_db():
#     conn = sqlite3.connect('vector_db.db')
#     cursor = conn.cursor()
#
#     cursor.execute('CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, data BLOB)')
#     conn.commit()
#     conn.close()
#
# def save_db():
#     conn = sqlite3.connect('vector_db.db')
#     cursor = conn.cursor()
#     data = np.array([1, 2, 3, 4, 5])
#     data_bytes = data.tobytes()
#
#     cursor.execute('INSERT INTO vectors (data) VALUES (?)', (data_bytes,))
#     conn.commit()
#     conn.close()
#
# def save_many_db():
#     conn = sqlite3.connect('vector_db.db')
#     cursor = conn.cursor()
#     arrays_to_insert = [np.array([1, 2, 3]),
#                         np.array([4, 5, 6]),
#                         np.array([7, 8, 9])]
#     data_bytes = [(array.tobytes(),) for array in arrays_to_insert]
#     cursor.executemany('INSERT INTO vectors (data) VALUES (?)', data_bytes)
#
#     conn.commit()
#     conn.close()
#
# def retrieve_db():
#     conn = sqlite3.connect('vector_db.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT id, data FROM vectors WHERE id = ?', (1,))
#     row = cursor.fetchone()
#
#     if row is not None:
#         retrieved_index, retrieved_data = row[0], np.frombuffer(row[1], dtype=np.int64)
#         print('index', retrieved_index)
#         print('data', retrieved_data)
#     conn.close()
