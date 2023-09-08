import pytest
import numpy as np
from simple_vector_db.quantization.vector_quantizer import VectorQuantizer

sample_vectors = [
    np.array([0.1, 0.2, 0.3]),
    np.array([0.4, 0.5, 0.6]),
    np.array([0.7, 0.8, 0.9]),
    np.array([0.5, 0.7, 0.3])
]


@pytest.fixture
def sample_high_d_vectors():
    return [
        np.array([11.0, 4.0, 2.0, 1.0, 3.0, 5.0, 7.0, 2.0]),
        np.array([2.0, 4.0, 3.0, 10.0, 4.0, 5.0, 3.0, 2.0]),
        np.array([3.0, 11.0, 2.0, 1.0, 3.0, 6.0, 3.0, 1.0]),
        np.array([2.0, 6.0, 2.0, 1.0, 3.0, 5.0, 1.0, 2.0]),
    ]


@pytest.fixture
def test_vector():
    # Vector of dimension 8 for testing
    return np.array([2.0, 4.0, 2.0, 1.0, 3.0, 5.0, 3.0, 2.0])


@pytest.fixture()
def test_quantizer():
    return VectorQuantizer(m_chunks=4, k_centroids=8)


def test_split_vector_into_chunks_should_return_correct_chunks(test_vector, test_quantizer):
    # GIVEN
    expected_chunks = [[2.0, 4.0], [2.0, 1.0], [3.0, 5.0], [3.0, 2.0]]
    # WHEN
    chunks_result = test_quantizer.split_vector_into_chunks(test_vector)
    # THEN
    assert (np.array_equal(chunks_result, expected_chunks))


def test_split_vector_into_chunks_should_return_an_error_if_divisible(test_vector, caplog):
    # GIVEN
    quantizer = VectorQuantizer(m_chunks=5, k_centroids=8)
    # WHEN
    chunks_result = quantizer.split_vector_into_chunks(test_vector)
    # THEN
    assert f"The vector's dimension 8 is not divisible by 5" in caplog.text


def test_compute_clusters_on_subspace_should_return_correct_number_of_clusters(test_quantizer):
    # GIVEN
    expected_number_of_cluster = 2
    assert test_quantizer.nb_subspace_centroids == expected_number_of_cluster
    # WHEN
    centroids, predicted_clusters, codebook_labels = test_quantizer.compute_clusters_on_subspace(sample_vectors,
                                                                                                 subspace_index=0)
    # THEN
    assert len(centroids) == expected_number_of_cluster
    assert len(set(predicted_clusters)) == expected_number_of_cluster


def test_compute_clusters_on_subspace_should_return_correct_global_codebook_index(test_quantizer):
    # GIVEN
    expected_codebook_indexes_for_subspace_2 = [5, 5, 4, 5]
    # WHEN
    centroids, predicted_clusters, codebook_labels = test_quantizer.compute_clusters_on_subspace(sample_vectors,
                                                                                                 subspace_index=2)
    # THEN
    for i, expected_el in enumerate(expected_codebook_indexes_for_subspace_2):
        assert predicted_clusters[i] == expected_el


def test_compute_clusters_on_subspace_should_return_correct_codebook_labels(test_quantizer):
    # GIVEN
    expected_codebook_labels_for_subspace_2 = [4, 5]
    # WHEN
    centroids, predicted_clusters, codebook_labels = test_quantizer.compute_clusters_on_subspace(sample_vectors,
                                                                                                 subspace_index=2)
    # THEN
    for i, expected_el in enumerate(expected_codebook_labels_for_subspace_2):
        assert codebook_labels[i] == expected_el


def test_compute_clusters_should_add_to_codebook_correct_clusters_and_ids(test_quantizer):
    # GIVEN
    centroid_0_4 = np.array([0.7, 0.8, 0.9])
    centroid_0_5 = np.array([0.33333333, 0.46666667, 0.4])
    # WHEN
    centroids, predicted_clusters, codebook_labels = test_quantizer.compute_clusters_on_subspace(sample_vectors,
                                                                                                 subspace_index=2)
    res_codebook = test_quantizer.codebook
    # THEN
    for i, exp_el in enumerate(centroid_0_4):
        assert res_codebook[4][i] == pytest.approx(exp_el)

    for i, exp_el in enumerate(centroid_0_5):
        assert res_codebook[5][i] == pytest.approx(exp_el)


def test_quantize_vectors_should_return_quantized_vectors(sample_high_d_vectors):
    # TODO code review: test plus détaillé?
    # GIVEN
    quantizer = VectorQuantizer(m_chunks=4, k_centroids=8)
    expected_shape = (4, 4) # 4 rows / 4 cols (one per chunk)
    # WHEN
    quantized_vectors = quantizer.quantize_vectors(sample_high_d_vectors)
    # THEN
    assert quantized_vectors.shape == expected_shape
    assert len(quantizer.codebook.keys()) == 8

if __name__ == "__main__":
    pytest.main()
