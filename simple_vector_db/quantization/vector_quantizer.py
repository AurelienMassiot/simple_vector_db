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
        self.codebook: Dict[int,dict[int,np.array]] = {}

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

    def quantize_vectors(self, input_vectors: list[np.array]):
        input_vectors_matrix = np.array(input_vectors)
        vector_dimension = input_vectors[0].shape[0]
        if vector_dimension % self.m_chunks != 0:
            logger.error(f"The vector's dimension {vector_dimension} is not divisible by {self.m_chunks}")
        chunks = np.split(input_vectors_matrix, self.m_chunks, axis=1)
        quantized_vector = []
        for i, chunk in enumerate(chunks):
            centroids, predicted_clusters = self.compute_clusters_on_subspace(chunk, i)
            quantized_vector.append(predicted_clusters)

        return np.array(quantized_vector).T

    def rebuild_vector(self, input_vector: np.array):
        rebuilt_vector = np.array([])
        for subspace_index,chunk_centroid_id in enumerate(input_vector):
            rebuilt_chunk = self.codebook[subspace_index][chunk_centroid_id]
            rebuilt_vector = np.append(rebuilt_vector, rebuilt_chunk)
        return rebuilt_vector


    def get_codebook_for_chunk_index(self,chunk_index):
        pass
    def compute_assymetric_distance_matrix(self):
        pass

    @staticmethod
    def distance_chunk_centroids(chunk: np.array, subspace_centroids: dict[int,np.array] ) -> np.array:
        return [euclidean_distance(chunk, centroid) for centroid in subspace_centroids.values()]
