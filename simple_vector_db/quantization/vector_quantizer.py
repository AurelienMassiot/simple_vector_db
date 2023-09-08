import numpy as np


class VectorQuantizer:

    def __init__(self, m_chunks: int, k_centroids: int):
        self.m_chunks = m_chunks
        self.k_centroids = k_centroids

    def split_vector_into_chunks(self, input_vector: np.array):
        return input_vector
