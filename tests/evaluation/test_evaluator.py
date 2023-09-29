from simple_vector_db.evaluation.evaluator import Dataset, VectorDBEvaluator
import numpy as np
from simple_vector_db.vector_db_sqlite import VectorDBSQLite
import shutil
import pytest

sample_vectors = [
    np.array([10, 0.1, 0.1]),
    np.array([0.11, 0.11, 0.11]),
    np.array([0.115, 0.115, 11]),
]


@pytest.fixture
def vector_db():
    db_filename = "test_db.db"
    db = VectorDBSQLite(db_filename)
    yield db
    shutil.os.remove(db_filename)


@pytest.fixture()
def dataset():
    ds = Dataset(sample_vectors, k=1)
    return ds


def test_data_set_creation():
    # given/when
    ds = Dataset(sample_vectors, k=1)
    # then
    assert ds.ids == [0, 1, 2]
    expected_knn = {0: [0,1], 1: [1,2], 2: [2,1]}
    for id_vect in ds.ids_to_brute_knn.keys():
        assert ds.ids_to_brute_knn[id_vect] == expected_knn[id_vect]


def test_evaluator_query_with_all_vectors(vector_db, dataset):
    # given
    query_vector = np.array([0.1, 0.2, 0.3])
    # when
    eval = VectorDBEvaluator(vector_db, dataset)
    # then
    results = eval.query_with_all_vectors(k=3)
    for res in results:
        assert len(res) == 3


def test_evaluator_compute_recall_on_results(vector_db):
    # given
    ds = Dataset(sample_vectors, k=2)
    eval = VectorDBEvaluator(vector_db, ds)

    # when
    results = eval.query_with_all_vectors(k=3)
    map = eval.compute_recall_on_results(results)

    assert map == 1.0
