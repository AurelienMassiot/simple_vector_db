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

quantizer = VectorQuantizer(m_chunks=8, k_centroids=128)


def perform_quantization():
    # les données MNIST c'est cool comme premier test
    vectors_to_quantize = load_digits().data
    quantized_vectors = quantizer.quantize_vectors(vectors_to_quantize)
    logger.info(f"First quantized vector (centroids ids): {quantized_vectors[0]}")
    codebook = quantizer.codebook
    logger.info(f"Current Codebook: {codebook}")
    logger.info(f"First original vector: {vectors_to_quantize[0]}")
    rebuilt_vector = quantizer.rebuild_vector(quantized_vectors[0])
    logger.info(f"First rebuilt vector: {rebuilt_vector}")
    compression_mse = mean_squared_error(vectors_to_quantize[0],rebuilt_vector)
    logger.info(f"Mean squared error of compression for first vector {compression_mse}")


if __name__ == "__main__":
    perform_quantization()
