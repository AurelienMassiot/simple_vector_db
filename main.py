import logging

import numpy as np

from simple_vector_db.vector_db import VectorDBInMemory, VectorDBSQLite
from utils.flex_logging import ch

logger = logging.getLogger(__name__)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def perform_search_in_memory():
    # Create an instance of the VectorDatabase
    vector_db = VectorDBInMemory()

    # Insert vectors into the database
    arrays_to_insert = [np.array([1, 2, 3]),
                        np.array([4, 5, 6]),
                        np.array([7, 8, 9])]
    vector_db.insert(arrays_to_insert)

    # Search for similar vectors
    query_vector = np.array([0.15, 0.25, 0.35])
    k_similar_vectors = 5
    similar_vectors = vector_db.search(query_vector, k=k_similar_vectors)
    logger.info(f"Most {k_similar_vectors} Similar vectors: {similar_vectors}")

    # Retrieve a specific vector by its key
    retrieved_vector = vector_db.retrieve(2)
    logger.info(f"Retrieved vectors: {retrieved_vector}")


def perform_search_sqlite():
    vector_db = VectorDBSQLite()

    # arrays_to_insert = [np.array([1, 2, 3]),
    #                     np.array([4, 5, 6]),
    #                     np.array([7, 8, 9])]
    num_arrays = 1000000
    array_shape = (1, 3)
    arrays_to_insert = [np.random.rand(*array_shape) for _ in range(num_arrays)]

    # Insert vectors into the database
    vector_db.insert(arrays_to_insert)

    # Search for similar vectors
    query_vector = np.array([0.15, 0.25, 0.35])
    k_similar_vectors = 5
    similar_vectors = vector_db.search(query_vector, k=k_similar_vectors)
    logger.info(f"Most {k_similar_vectors} Similar vectors: {similar_vectors}")

    # Retrieve a specific vector by its key
    retrieved_vector = vector_db.retrieve(1)
    logger.info(f"Retrieved vectors: {retrieved_vector}")


if __name__ == "__main__":
    #perform_search_in_memory()
    perform_search_sqlite()
