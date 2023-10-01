import sqlite3
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

from simple_vector_db.distances import cosine_similarity, euclidean_distance
from simple_vector_db.numpy_array_adapter import NumpyArrayAdapter

Base = declarative_base()


class Vector(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True)
    data = Column(NumpyArrayAdapter)

    def __init__(self, data):
        self.data = data



class Mapping(Base):
    __tablename__ = "id_map"

    id = Column(Integer, primary_key=True)
    external_id = Column(Integer)

    def __init__(self, id, external_id):
        self.id = id
        self.external_id = external_id


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
        self.metric = {"metric": cosine_similarity, "reverse_sort": True}

    def set_metric(self, metric_name: str = "cosine"):
        if metric_name == "cosine":
            self.metric["metric"] = cosine_similarity
            self.metric["reverse_sort"] = True
        elif metric_name == "euclidean":
            self.metric["metric"] = euclidean_distance
            self.metric["reverse_sort"] = False

    def insert(self, vectors: list[np.ndarray], vector_ids: list[int] = None) -> None:
        vector_objects = [Vector(data=array) for array in vectors]
        self.session.add_all(vector_objects)
        self.session.commit()
        if vector_ids is not None:
            self.create_id_map(vector_objects, vector_ids)

    def create_id_map(self, inserted_objects: list[Vector], vector_ids: list[int]):
        mapping_to_insert = [Mapping(vector.id, ext_id) for vector, ext_id in zip(inserted_objects, vector_ids)]
        self.session.add_all(mapping_to_insert)
        self.session.commit()

    def search_mapping_id(self, external_id: int):
        result_mapping_id = self.session.query(Mapping).filter_by(external_id=external_id).all()
        ids = [mapping.id for mapping in result_mapping_id]
        return ids

    def search_without_index(
            self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[int, float]]:
        vectors = self.session.query(Vector).all()
        similarities = [
            (vector.id, self.metric["metric"](query_vector, vector.data))
            for vector in vectors
        ]
        similarities.sort(key=lambda x: x[1], reverse=self.metric["reverse_sort"])
        top_similarities = similarities[:k]

        return top_similarities

    def search_with_id(self, query_id: int):
        vectors = self.session.query(Vector).filter_by(id=query_id).all()
        if len(vectors) > 0:
            return vectors[0]
        else:
            return None

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
        most_similar_centroid_id = self.find_most_similar_centroid(
            query_vector, centroids
        )

        indexed_vectors = (
            self.session.query(IndexedVector)
            .filter_by(cluster=most_similar_centroid_id)
            .all()
        )

        similarities = [
            (indexed_vector.id, self.metric["metric"](query_vector, indexed_vector.data))
            for indexed_vector in indexed_vectors
        ]

        similarities.sort(key=lambda x: x[1], reverse=self.metric["reverse_sort"])
        most_similar_vectors = similarities[:k]

        return most_similar_vectors, most_similar_centroid_id

    def find_most_similar_centroid(self, query_vector, centroids) -> int:
        centroid_similarities = [
            (centroid.id, self.metric["metric"](query_vector, centroid.data))
            for centroid in centroids
        ]
        centroid_similarities.sort(key=lambda x: x[1], reverse=self.metric["reverse_sort"])
        most_similar_centroid_id = centroid_similarities[0][0]
        return most_similar_centroid_id

    def insert_centroids(self, centroids) -> None:
        centroid_objects = [
            Centroid(id=id, data=centroid_coordinates)
            for id, centroid_coordinates in enumerate(centroids)  # because centroids id have to start at O
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
