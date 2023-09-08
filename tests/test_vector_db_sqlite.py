import shutil

import numpy as np
import pytest

from simple_vector_db.vector_db_sqlite import VectorDBSQLite, Vector

sample_vectors = [
    np.array([0.1, 0.2, 0.3]),
    np.array([0.4, 0.5, 0.6]),
    np.array([0.7, 0.8, 0.9]),
]


@pytest.fixture
def vector_db():
    db_filename = "test_db.db"
    db = VectorDBSQLite(db_filename)
    yield db
    shutil.os.remove(db_filename)


def test_insert(vector_db):
    # Given When
    vector_db.insert(sample_vectors)

    # Then
    count = vector_db.session.query(Vector).count()
    assert count == len(sample_vectors)


def test_search_without_index(vector_db):
    # Given
    vector_db.insert(sample_vectors)
    query_vector = np.array([0.3, 0.4, 0.5])
    k = 2

    # When
    results = vector_db.search_without_index(query_vector, k)

    # Then
    assert len(results) == k
    for i in range(1, k):
        assert results[i][1] <= results[i - 1][1]


def test_create_kmeans_index(vector_db):
    # Given
    vector_db.insert(sample_vectors)
    n_clusters = 2

    # When
    centroids = vector_db.create_kmeans_index(n_clusters)

    # Then
    assert len(centroids) == n_clusters


def test_search_in_kmeans_index(vector_db):
    # Given
    vector_db.insert(sample_vectors)
    n_clusters = 2
    query_vector = np.array([0.3, 0.4, 0.5])
    k = 2
    vector_db.create_kmeans_index(n_clusters)

    # When
    results, most_similar_centroid = vector_db.search_in_kmeans_index(query_vector, k)

    # Then
    assert len(results) == k
    for i in range(1, k):
        assert results[i][1] <= results[i - 1][1]
    assert isinstance(most_similar_centroid, int)


if __name__ == "__main__":
    pytest.main()
