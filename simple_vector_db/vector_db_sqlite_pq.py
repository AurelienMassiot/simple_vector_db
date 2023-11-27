import numpy as np
from sklearn.datasets import load_digits
from simple_vector_db.quantization.vector_quantizer import VectorQuantizer
from simple_vector_db.vector_db_sqlite import VectorDBSQLite, Vector


class VectorDBSQLitePQ(VectorDBSQLite):

    def __init__(self, db_filename: str, m_chunks: int, nb_subspace_centroids: int):
        super(VectorDBSQLitePQ, self).__init__(db_filename)
        self.db_quantizer = VectorQuantizer(m_chunks, nb_subspace_centroids)

    def insert_pq(self, vectors: list[np.ndarray], vector_ids: list[int] = None):
        quantized_vectors: list[np.ndarray] = self.db_quantizer.quantize_vectors(vectors)
        self.insert(quantized_vectors, vector_ids)

    def search_with_pq(self, query_vector: np.ndarray, k: int):
        quantized_vectors = self.session.query(Vector).all()
        quantized_vectors_data = [vec.data for vec in quantized_vectors]
        quantized_vectors_ids = [vec.id for vec in quantized_vectors]
        distance_matrix = self.db_quantizer.compute_assymetric_distance_matrix(query_vector)
        distances = self.db_quantizer.compute_distances_for_all_vectors(distance_matrix, quantized_vectors_data)
        #distances_ids = list(zip(distances,quantized_vectors_ids))
        distances_ids = list(zip(distances, range(len(distances))))
        ann_results = sorted(distances_ids[0:k], key=lambda x: x[0])
        return ann_results


if __name__ == "__main__":
    vectordbPQ = VectorDBSQLitePQ("vector_db.db", m_chunks=16, nb_subspace_centroids=32)
    vectors_to_quantize = load_digits().data
    labels = load_digits().target
    vectordbPQ.insert_pq(vectors_to_quantize, list(range(0, len(vectors_to_quantize))))
    idx_query_vector = 11
    query_vector = vectors_to_quantize[idx_query_vector]
    print(vectordbPQ.search_with_pq(query_vector, k=10))

