import shutil

import numpy as np
import pytest

from simple_vector_db.vector_db_sqlite import VectorDBSQLite


@pytest.fixture
def vector_db():
    db = VectorDBSQLite()
    yield db
    db.conn.close()
    shutil.os.remove('vector_db.db')


def test_insert_and_retrieve(vector_db):
    #  Given
    vectors_to_insert = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

    # When
    vector_db.insert(vectors_to_insert)
    retrieved_index, retrieved_data = vector_db.retrieve(1)

    # Then
    assert retrieved_index == 1
    assert np.array_equal(retrieved_data, np.array([1, 2, 3]))


def test_search_with_a_similar_vector_should_return_an_ordered_list_of_vectors_with_first_similarity_equal_to_1(
        vector_db):
    #  Given
    vectors_to_insert = [np.array([1, 2, 3]), np.array([3, 2, 1])]
    vector_db.insert(vectors_to_insert)
    query_vector = np.array([1, 2, 3])
    k = 2

    # When
    similar_vectors = vector_db.search(query_vector, k)

    # Then
    assert len(similar_vectors) == k
    assert similar_vectors[0][0] == 1
    assert similar_vectors[0][1] == 1.0


def test_retrieve_nonexistent_key_should_return_none(vector_db):
    # Given
    nonexistent_key = 100

    # When
    retrieved_vector = vector_db.retrieve(nonexistent_key)

    # Then
    assert retrieved_vector is None
