import pytest
import numpy as np
from simple_vector_db.quantization.vector_quantizer import VectorQuantizer


@pytest.fixture
def test_vector():
    # Vector of dimension 8 for testing
    return np.array([2.0, 4.0, 2.0, 1.0, 3.0, 5.0, 3.0, 2.0])


def test_split_vector_into_chunks_should_return_correct_chunks(test_vector):
    # GIVEN
    expected_chunks = [[2.0, 4.0], [2.0, 1.0], [3.0, 5.0], [3.0, 2.0]]
    quantizer = VectorQuantizer(m_chunks=4, k_centroids=4)
    # WHEN
    chunks_result = quantizer.split_vector_into_chunks(test_vector)
    # THEN
    assert (np.array_equal(chunks_result, expected_chunks))


def test_split_vector_into_chunks_should_return_an_error_if_divisible(test_vector, caplog):
    # GIVEN
    quantizer = VectorQuantizer(m_chunks=5, k_centroids=4)
    # WHEN
    chunks_result = quantizer.split_vector_into_chunks(test_vector)
    # THEN
    assert f"The vector's dimension 8 is not divisible by 5" in caplog.text
