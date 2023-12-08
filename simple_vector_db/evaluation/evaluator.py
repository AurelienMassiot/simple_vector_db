import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from simple_vector_db.vector_db import VectorDB
from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


class Dataset:
    def __init__(self, vectors, k):
        self.vectors: list[np.array] = vectors
        self.k = k
        self.ids: list[int] = None
        self.brute_knn: NearestNeighbors = None
        self.ids_to_brute_knn: dict = {}
        self.ids_to_distance_brute_knn: dict = {}
        self.set_ids()
        self.init_brute_force_knn()
        self.map_ids_to_brute_knn()

    def set_ids(self):
        self.ids = list(range(len(self.vectors)))

    def map_ids_to_brute_knn(self):
        for currid in self.ids:
            res_knn = self.brute_knn.kneighbors(self.vectors[currid].reshape(1, -1), return_distance=True)
            reel_distances = res_knn[0][0]
            reel_knn = list(res_knn[1][0])
            self.ids_to_brute_knn[currid] = reel_knn
            self.ids_to_distance_brute_knn[currid] = reel_distances

    def init_brute_force_knn(self):
        brute_knn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="brute", metric='euclidean')
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
        results = [self.vector_db.search_without_index(vector, k + 1) for vector in self.bench_dataset.vectors]
        return results

    def set_kmeans_index(self, nb_clusters: int):
        self.vector_db.create_kmeans_index(nb_clusters)

    def query_with_all_vectors_kmeans_index(self, k: int, n_probes=1):
        results = [self.vector_db.search_in_kmeans_index(vector, k + 1, n_probes=n_probes)[0] for vector in
                   self.bench_dataset.vectors]
        return results

    def compute_recall_on_results(self, results):
        total_recall = 0
        for query_id, res in enumerate(results):
            real_knn = self.bench_dataset.ids_to_brute_knn[query_id]
            pred_knn = [res_vec_id - 1 for res_vec_id, _ in res]
            intersection = set(real_knn).intersection(pred_knn)
            recall = len(intersection) / len(pred_knn)
            total_recall += recall
        return total_recall / len(results)
