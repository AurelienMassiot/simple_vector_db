import numpy as np
import logging
from utils.flex_logging import stream_handler
from sklearn.cluster import KMeans
from typing import Dict
from simple_vector_db.config import LOGGING_LEVEL
from simple_vector_db.distances import euclidean_distance

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(LOGGING_LEVEL)


class VectorQuantizer:

    def __init__(self, m_chunks: int, nb_subspace_centroids: int):
        self.m_chunks = m_chunks
        self.nb_subspace_centroids = nb_subspace_centroids
        self.codebook: Dict[int, dict[int, np.array]] = {}

    def split_vector_into_chunks(self, input_vector: np.array):
        vector_dimension = input_vector.shape[0]
        if vector_dimension % self.m_chunks != 0:
            logger.error(f"The vector's dimension {vector_dimension} is not divisible by {self.m_chunks}")
        chunk_dimension = int(vector_dimension / self.m_chunks)
        chunks = [input_vector[ch * chunk_dimension:(ch + 1) * chunk_dimension] for ch in range(self.m_chunks)]
        return chunks

    def compute_clusters_on_subspace(self, subspace_vectors: list[np.array], subspace_index: int):
        kmeans = KMeans(n_clusters=self.nb_subspace_centroids, random_state=0, n_init="auto")
        kmeans.fit_predict(subspace_vectors)
        centroids = kmeans.cluster_centers_
        predicted_clusters = kmeans.labels_
        subspace_codebook = {}
        for i, el in enumerate(centroids):
            subspace_codebook[i] = el
        self.codebook[subspace_index] = subspace_codebook
        return centroids, predicted_clusters

    def quantize_vectors(self, input_vectors: list[np.array]) -> list[np.array]:
        input_vectors_matrix = np.array(input_vectors)
        vector_dimension = input_vectors[0].shape[0]
        if vector_dimension % self.m_chunks != 0:
            logger.error(f"The vector's dimension {vector_dimension} is not divisible by {self.m_chunks}")
        chunks = np.split(input_vectors_matrix, self.m_chunks, axis=1)
        quantized_vector = [self.compute_clusters_on_subspace(chunk, i)[1] for i, chunk in enumerate(chunks)]
        return np.array(quantized_vector).T

    def rebuild_vector(self, input_vector: np.array):
        rebuilt_vector = np.array([])
        for subspace_index, chunk_centroid_id in enumerate(input_vector):
            rebuilt_chunk = self.codebook[subspace_index][chunk_centroid_id]
            rebuilt_vector = np.append(rebuilt_vector, rebuilt_chunk)
        return rebuilt_vector

    def compute_assymetric_distance_matrix(self, query_vector: np.array) -> dict[int, dict]:
        chunks = self.split_vector_into_chunks(query_vector)
        distance_matrix = {subspace_index: self.distance_chunk_centroids(chunk, self.codebook[subspace_index]) for
                           subspace_index, chunk in enumerate(chunks)}
        return distance_matrix

    def compute_distances_for_all_vectors(self, distance_matrix: dict[int, dict], quantized_vectors: list[np.array]):
        distance_list = []
        for vector in quantized_vectors:
            subspace_distances = np.array(
                [distance_matrix[int(subspace_id)][int(cluster_id)] for subspace_id, cluster_id in enumerate(vector)]) # casting in int because of a bug in numpy with float64
            distance_list.append(subspace_distances.sum())
        return distance_list

    @staticmethod
    def distance_chunk_centroids(chunk: np.array, subspace_centroids: dict[int, np.array]) -> np.array:
        return {centroid_id: euclidean_distance(chunk, subspace_centroids[centroid_id]) for centroid_id in
                subspace_centroids.keys()}
