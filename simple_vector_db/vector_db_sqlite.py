import sqlite3
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from simple_vector_db.distances import cosine_similarity
from simple_vector_db.numpy_array_adapter import NumpyArrayAdapter

Base = declarative_base()


class Vector(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True)
    data = Column(NumpyArrayAdapter)

    def __init__(self, data):
        self.data = data


class Centroid(Base):
    __tablename__ = "centroids"

    id = Column(Integer, primary_key=True)
    data = Column(NumpyArrayAdapter)

    def __init__(self, id, data):
        self.id = id
        self.data = data


class IndexedVector(Base):
    __tablename__ = "indexed_vectors"

    id = Column(Integer, primary_key=True)
    data = Column(NumpyArrayAdapter)
    cluster = Column(Integer)

    def __init__(self, data, cluster):
        self.data = data
        self.cluster = cluster


class VectorDBSQLite:
    def __init__(self, db_filename: str):
        self.engine = create_engine(f"sqlite:///{db_filename}", connect_args={'detect_types': sqlite3.PARSE_DECLTYPES})
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def insert(self, vectors: list[np.ndarray]) -> None:
        vector_objects = [Vector(data=array) for array in vectors]
        self.session.add_all(vector_objects)
        self.session.commit()

    def search_without_index(
            self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[int, float]]:
        vectors = self.session.query(Vector).all()

        similarities = [
            (vector.id, cosine_similarity(query_vector, vector.data))
            for vector in vectors
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:k]

        return top_similarities

    def create_kmeans_index(self, n_clusters: int) -> np.ndarray:
        vectors = self.session.query(Vector).all()
        vector_arrays = [vector.data for vector in vectors]

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit_predict(vector_arrays)
        centroids = kmeans.cluster_centers_

        self.insert_centroids(centroids)
        self.insert_indexed_vectors(vector_arrays, kmeans.labels_)

        return centroids

    def search_in_kmeans_index(
            self, query_vector: np.ndarray, k: int
    ) -> Tuple[List[Tuple[int, float]], int]:
        centroids = self.session.query(Centroid).all()
        most_similar_centroid = self.find_most_similar_centroid(
            query_vector, centroids
        )

        indexed_vectors = (
            self.session.query(IndexedVector)
                .filter_by(cluster=most_similar_centroid)
                .all()
        )

        similarities = [
            (indexed_vector.id, cosine_similarity(query_vector, indexed_vector.data))
            for indexed_vector in indexed_vectors
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        most_similar_vectors = similarities[:k]

        return most_similar_vectors, most_similar_centroid

    def find_most_similar_centroid(self, query_vector, centroids) -> int:
        centroid_similarities = [
            (centroid.id, cosine_similarity(query_vector, centroid.data))
            for centroid in centroids
        ]
        centroid_similarities.sort(key=lambda x: x[1], reverse=True)
        most_similar_centroid_id = centroid_similarities[0][0]
        return most_similar_centroid_id

    def insert_centroids(self, centroids) -> None:
        centroid_objects = [
            Centroid(id=id, data=centroid_coordinates)
            for id, centroid_coordinates in enumerate(centroids) # because centroids id have to start at O
        ]
        self.session.add_all(centroid_objects)
        self.session.commit()

    def insert_indexed_vectors(self, vectors, clusters) -> None:
        indexed_vector_objects = [
            IndexedVector(
                data=vector_coordinates, cluster=int(cluster)
            )
            for vector_coordinates, cluster in zip(vectors, clusters)
        ]
        self.session.add_all(indexed_vector_objects)
        self.session.commit()
