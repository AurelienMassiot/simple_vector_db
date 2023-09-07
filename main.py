import logging

import numpy as np

from simple_vector_db.vector_db import VectorDBInMemory, VectorDBSQLite
from utils.flex_logging import stream_handler
from utils.timing import timeit

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
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

    num_arrays = 10000
    array_shape = (1, 3)
    arrays_to_insert = [np.random.rand(*array_shape) for _ in range(num_arrays)]

    # Insert vectors into the database
    vector_db.insert(arrays_to_insert)
    print('avant', arrays_to_insert[0])

    # Search for similar vectors
    query_vector = np.array([1, 2, 3])
    k_similar_vectors = 5
    similar_vectors = vector_db.search(query_vector, k=k_similar_vectors)
    logger.info(f"Most {k_similar_vectors} Similar vectors: {similar_vectors}")

    # Retrieve a specific vector by its key
    retrieved_vector = vector_db.retrieve(1)
    logger.info(f"Retrieved vectors: {retrieved_vector}")


@timeit
def perform_search_only_sqlite():
    vector_db = VectorDBSQLite()
    query_vector = np.array([0.15, 0.25, 0.35])
    k_similar_vectors = 5
    similar_vectors = vector_db.search(query_vector, k=k_similar_vectors)
    logger.info(f"Most {k_similar_vectors} Similar vectors: {similar_vectors}")


def reindex_vectors():
    vector_db = VectorDBSQLite()
    df_vectors, df_centroids = vector_db.index(n_clusters=3)
    logger.info(df_vectors)
    logger.info(df_centroids)


if __name__ == "__main__":
    # perform_search_in_memory()
    #perform_search_sqlite()
    # perform_search_only_sqlite()
    reindex_vectors()
