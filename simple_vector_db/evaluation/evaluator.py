import numpy as np
from simple_vector_db.vector_db import VectorDB
from sklearn.neighbors import NearestNeighbors
import logging
from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

class Dataset:
    def __init__(self, vectors, k):
        self.vectors: list[np.array] = vectors
        self.k = k
        self.ids: list[int] = None
        self.brute_knn: NearestNeighbors = None
        self.ids_to_brute_knn: dict = {}
        self.set_ids()
        self.init_brute_force_knn()
        self.map_ids_to_brute_knn()

    def set_ids(self):
        self.ids = list(range(len(self.vectors)))

    def map_ids_to_brute_knn(self):
        for currid in self.ids:
            reel_knn = list(self.brute_knn.kneighbors(self.vectors[currid].reshape(1, -1))[1][0])
            self.ids_to_brute_knn[currid] = reel_knn

    def init_brute_force_knn(self):
        brute_knn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="brute", metric='cosine')
        brute_knn.fit(self.vectors)
        self.brute_knn = brute_knn


class VectorDBEvaluator:

    def __init__(self, vector_db_to_benchmark, bench_dataset: Dataset):
        self.vector_db: VectorDB = vector_db_to_benchmark
        self.bench_dataset: Dataset = bench_dataset
        self.insert_dataset_to_vector_db()

    def insert_dataset_to_vector_db(self):
        vectors = self.bench_dataset.vectors
        ids = self.bench_dataset.ids
        self.vector_db.insert(vectors, vector_ids=ids)

    def query_with_all_vectors(self, k: int):
        results = []
        for vector in self.bench_dataset.vectors:
            results.append(self.vector_db.search_without_index(vector, k+1))
        return results

    def compute_recall_on_results(self, results):
        total_recall = 0
        for query_id, res in enumerate(results):
            real_knn = self.bench_dataset.ids_to_brute_knn[query_id]
            pred_knn = []
            for res_vec_id, sim in res:
                pred_knn.append(res_vec_id-1)
            real_knn.sort()
            pred_knn.sort()

            recall = len(set(real_knn).intersection(pred_knn)) / len(pred_knn)
            if recall < 1:
                logger.debug(real_knn, "/", pred_knn)
                logger.debug(results[query_id])
            total_recall += recall
        return total_recall / len(results)
