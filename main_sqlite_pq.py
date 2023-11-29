import logging
import os

from sklearn.datasets import load_digits

from simple_vector_db.vector_db_sqlite_pq import VectorDBSQLitePQ
from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

DB_FILENAME = "vector_db.db"
IDX_QUERY_VECTOR = 11
K_SIMILAR_VECTORS = 10
M_CHUNKS = 16
N_CENTROIDS = 32


def clean_sqlite_file_if_exists():
    try:
        os.remove(DB_FILENAME)
    except FileNotFoundError:
        pass


def perform_query_on_quantized_db():
    vectordbPQ = VectorDBSQLitePQ(DB_FILENAME, m_chunks=M_CHUNKS, nb_subspace_centroids=N_CENTROIDS)
    vectors_to_quantize = load_digits().data
    vectordbPQ.insert_pq(vectors_to_quantize, list(range(0, len(vectors_to_quantize))))
    idx_query_vector = 11
    query_vector = vectors_to_quantize[idx_query_vector]
    results = vectordbPQ.search_with_pq(query_vector, k=K_SIMILAR_VECTORS)
    logger.info("Results Vector are:" + str(results))


if __name__ == "__main__":
    clean_sqlite_file_if_exists()
    perform_query_on_quantized_db()
