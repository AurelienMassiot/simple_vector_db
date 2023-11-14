import logging

import numpy as np
from sklearn.datasets import load_digits
from simple_vector_db.quantization.vector_quantizer import VectorQuantizer
from utils.flex_logging import stream_handler
from simple_vector_db.config import LOGGING_LEVEL
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(LOGGING_LEVEL)

quantizer = VectorQuantizer(m_chunks=16, nb_subspace_centroids=32)


def perform_quantization():
    # les données MNIST c'est cool comme premier test
    vectors_to_quantize = load_digits().data
    labels = load_digits().target
    idx_query_vector = 11
    query_vector = vectors_to_quantize[idx_query_vector]
    quantized_vectors = quantizer.quantize_vectors(vectors_to_quantize)
    logger.info(f"quantized vector (centroids ids): {quantized_vectors[idx_query_vector]}")
    #codebook = quantizer.codebook
    #logger.info(f"Current Codebook: {codebook}")
    logger.info(f"original vector: {query_vector}")
    rebuilt_vector = quantizer.rebuild_vector(quantized_vectors[idx_query_vector])
    logger.info(f"rebuilt vector: {rebuilt_vector}")
    compression_mse = mean_squared_error(query_vector, rebuilt_vector)
    logger.info(f"Mean squared error of compression for first vector {compression_mse}")
    knn = find_knn_with_quantization(query_vector, quantizer, quantized_vectors)
    logger.info(f'Label for the query vector is {labels[idx_query_vector]}')
    nn_ids = [tuple[1] for tuple in knn]
    nn_labels = labels[nn_ids]
    logger.info(f'Labels of the 20 nearest neighbors are {nn_labels}')


def find_knn_with_quantization(query_vector: np.array, quantizer: VectorQuantizer,
                               quantized_vectors: list[np.array], k: int = 20):
    distance_matrix = quantizer.compute_assymetric_distance_matrix(query_vector)
    distances = quantizer.compute_distances_for_all_vectors(distance_matrix, quantized_vectors)
    distances_ids = list(zip(distances, range(len(distances))))
    return sorted(distances_ids[0:k])


if __name__ == "__main__":
    perform_quantization()
