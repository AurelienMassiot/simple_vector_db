import logging

import numpy as np

from simple_vector_db.vector_db_in_memory import VectorDBInMemory
from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

vector_db = VectorDBInMemory()
QUERY_VECTOR = np.array([10, 20, 30])
K_SIMILAR_VECTORS = 3


def perform_search_in_memory():
    vectors_to_insert = [np.array([10, 20, 30]),
                         np.array([-1, -2, -3]),
                         np.array([0.3, 0.3, 0.3])]
    vector_db.insert(vectors_to_insert)

    retrieved_vector = vector_db.retrieve(1)
    logger.info(f"Retrieved vectors: {retrieved_vector}")

    similar_vectors = vector_db.search(query_vector=QUERY_VECTOR, k=K_SIMILAR_VECTORS)
    logger.info(f"Most {K_SIMILAR_VECTORS} Similar vectors: {similar_vectors}")


if __name__ == "__main__":
    perform_search_in_memory()
