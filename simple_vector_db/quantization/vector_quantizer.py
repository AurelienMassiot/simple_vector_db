import numpy as np
import logging
from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class VectorQuantizer:

    def __init__(self, m_chunks: int, k_centroids: int):
        self.m_chunks = m_chunks
        self.k_centroids = k_centroids

    def split_vector_into_chunks(self, input_vector: np.array):
        vector_dimension = input_vector.shape[0]
        if vector_dimension % self.m_chunks != 0:
            logger.error(f"The vector's dimension {vector_dimension} is not divisible by {self.m_chunks}")
        chunck_dimension = int(vector_dimension / self.m_chunks)
        chunks = [input_vector[ch * chunck_dimension:(ch + 1) * chunck_dimension] for ch in range(self.m_chunks)]
        return chunks

    def compute_clusters_on_subspace(self):

