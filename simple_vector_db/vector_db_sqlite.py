from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, Column, Integer, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from simple_vector_db.distances import cosine_similarity

Base = declarative_base()


class Vector(Base):
    __tablename__ = 'vectors'

    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)

    def __init__(self, data):
        self.data = data


class Centroid(Base):
    __tablename__ = 'centroids'

    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)

    def __init__(self, data):
        self.data = data


class IndexedVector(Base):
    __tablename__ = 'indexed_vectors'

    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    cluster = Column(Integer)

    def __init__(self, data, cluster):
        self.data = data
        self.cluster = cluster


class VectorDBSQLite:
    def __init__(self, db_filename: str):
        self.engine = create_engine(f'sqlite:///{db_filename}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def insert(self, vectors: list[np.ndarray]) -> None:
        vector_objects = [Vector(data=array.tobytes()) for array in vectors]
        self.session.add_all(vector_objects)
        self.session.commit()

    def search_without_index(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        vectors = self.session.query(Vector).all()
        vector_data = [(vector.id, self._bytes_to_array(vector.data)) for vector in vectors]

        similarities = [(idx, cosine_similarity(query_vector, vector_data[idx][1])) for idx in
                        range(len(vector_data))]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:k]

        result = [(vector_data[idx][0], similarity) for idx, similarity in top_similarities]
        return result

    def create_kmeans_index(self, n_clusters: int):
        vectors = self.session.query(Vector).all()
        vector_data = [self._bytes_to_array(vector.data) for vector in vectors]

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit_predict(vector_data)

        centroids = kmeans.cluster_centers_
        self.insert_centroids(centroids)
        self.insert_indexed_vectors(vector_data, kmeans.labels_)

        return centroids

    def search_in_kmeans_index(
            self, query_vector: np.ndarray, k: int
    ) -> Tuple[List[Tuple[int, float]], int]:
        centroids = self.session.query(Centroid).all()
        centroid_data = [np.frombuffer(centroid.data, dtype=np.float64) for centroid in centroids]
        most_similar_centroid = self.find_most_similar_centroid(query_vector, centroid_data)

        indexed_vectors = self.session.query(IndexedVector).filter_by(cluster=most_similar_centroid).all()
        indexed_vector_data = [(vector.id, self._bytes_to_array(vector.data)) for vector in indexed_vectors]

        similarities = [(vector[0], cosine_similarity(query_vector, vector[1])) for vector in indexed_vector_data]

        similarities.sort(key=lambda x: x[1], reverse=True)
        most_similar_vectors = similarities[:k]

        return most_similar_vectors, most_similar_centroid

    def find_most_similar_centroid(self, query_vector, centroids):
        centroid_similarities = [
            (idx, cosine_similarity(query_vector, centroid_data))
            for idx, centroid_data in enumerate(centroids)
        ]
        centroid_similarities.sort(key=lambda x: x[1], reverse=True)
        most_similar_centroid = centroid_similarities[0][0]
        return most_similar_centroid

    def insert_centroids(self, centroids):
        centroid_objects = [Centroid(data=self._array_to_bytes(centroid_coordinates)) for centroid_coordinates in
                            centroids]
        self.session.add_all(centroid_objects)
        self.session.commit()

    def insert_indexed_vectors(self, vectors, clusters):
        indexed_vector_objects = [
            IndexedVector(data=self._array_to_bytes(vector_coordinates), cluster=int(cluster))
            for vector_coordinates, cluster in zip(vectors, clusters)
        ]
        self.session.add_all(indexed_vector_objects)
        self.session.commit()

    def _array_to_bytes(self, array: np.ndarray) -> bytes:
        return array.tobytes()

    def _bytes_to_array(self, bytes: bytes) -> np.ndarray:
        return np.frombuffer(bytes, dtype=np.float64)
