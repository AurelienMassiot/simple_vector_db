import logging

import numpy as np

from simple_vector_db.vector_db_sqlite import VectorDBSQLite
from utils.flex_logging import stream_handler
from utils.timing import timeit

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

vector_db = VectorDBSQLite()
N_VECTORS = 10000
VECTORS_SHAPE = (1, 3)
QUERY_VECTOR = np.array([0.15, 0.25, 0.35])
K_SIMILAR_VECTORS = 5

N_CLUSTERS = 3


def insert_vectors(n_vectors: int, vectors_shape: tuple[int, int]) -> None:
    vectors_to_insert = [np.random.rand(*vectors_shape) for _ in range(n_vectors)]
    vector_db.insert(vectors_to_insert)


@timeit
def perform_search_without_index():
    similar_vectors = vector_db.search(QUERY_VECTOR, k=K_SIMILAR_VECTORS)
    logger.info(f"Most {K_SIMILAR_VECTORS} Similar vectors: {similar_vectors}")


def create_index():
    df_vectors = vector_db.create_index_kmeans(n_clusters=N_CLUSTERS)
    logger.info(df_vectors)


@timeit
def perform_search_with_index():
    (
        most_similar_vectors,
        most_similar_centroid,
        centroid_similarities,
    ) = vector_db.search_in_kmeans_index(query_vector=QUERY_VECTOR, k=K_SIMILAR_VECTORS)
    logger.info(f"Most {K_SIMILAR_VECTORS} Similar vectors: {most_similar_vectors}")
    logger.info(f"Most similar centroid: {most_similar_centroid}")
    logger.info(f"Centroid similarities: {centroid_similarities}")


if __name__ == "__main__":
    insert_vectors(N_VECTORS, VECTORS_SHAPE)
    create_index()
    perform_search_without_index()
    perform_search_with_index()
