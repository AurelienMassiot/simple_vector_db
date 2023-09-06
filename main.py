import logging

import numpy as np

from simple_vector_db.vector_db import VectorDatabase
from utils.flex_logging import ch

logger = logging.getLogger(__name__)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def perform_search():
    # Create an instance of the VectorDatabase
    vector_db = VectorDatabase()

    # Insert vectors into the database
    vector_db.insert("vector_1", np.array([0.2, 0.2, 0.2]))
    vector_db.insert("vector_2", np.array([0.4, 0.4, 0.4]))
    vector_db.insert("vector_3", np.array([0.6, 0.6, 0.6]))

    # Search for similar vectors
    query_vector = np.array([0.15, 0.25, 0.35])
    similar_vectors = vector_db.search(query_vector, k=2)
    logger.info(f"Similar vectors: {similar_vectors}")

    # Retrieve a specific vector by its key
    retrieved_vector = vector_db.retrieve("vector_1")
    logger.info(f"Retrieved vectors: {retrieved_vector}")


if __name__ == "__main__":
    perform_search()
