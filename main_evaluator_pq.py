import random
import shutil

import pandas as pd

from simple_vector_db.evaluation.evaluator import Dataset, VectorDBEvaluator
from simple_vector_db.vector_db_sqlite_pq import VectorDBSQLitePQ
from sklearn.datasets import load_digits
import logging
from utils.flex_logging import stream_handler
import dataget
import numpy as np
import time
from utils import plotting

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


def load_mnist_data():
    digits = load_digits().data
    random.shuffle(digits)
    # digits = digits[0:100]
    return digits


def load_fashion_mnist(n_sample=1000):
    X_train, y_train, X_test, y_test = dataget.image.fashion_mnist().get()
    X = np.vstack([X_train, X_test])
    images = []
    for image in X:
        images.append(image.flatten())
    random.shuffle(images)
    images = images[0:n_sample]
    return images


def remove_sqlite_file():
    db_filename = "test_db.db"
    try:
        shutil.os.remove(db_filename)
    except FileNotFoundError:
        pass


def prepare_sql_db(m_chunks: int, n_centroids: int):
    db_filename = "test_db.db"
    remove_sqlite_file()
    vector_db = VectorDBSQLitePQ(db_filename, m_chunks=m_chunks, nb_subspace_centroids=n_centroids)
    vector_db.set_metric("euclidean")
    return vector_db


if __name__ == "__main__":
    remove_sqlite_file()
    images = load_fashion_mnist(n_sample=1000)
    logger.info("Loaded Dataset")
    logger.info("Initiated Vector DB")
    results_bench = []
    for m_chunks in range(2, 64, 2):
        for n_centroids in range(2, 100, 5):
            vector_db = prepare_sql_db(m_chunks=m_chunks, n_centroids=n_centroids)
            k = 10
            ds = Dataset(images, k=k)
            try:
                eval = VectorDBEvaluator(vector_db, ds)
            except ValueError:
                logger.info("Skipping this iteration.")
                continue
            logger.info("Created evaluation Dataset")
            time_start = time.time()
            results = eval.query_with_all_vectors(k=k)
            time_end = time.time()
            time_total = time_end - time_start
            recall = eval.compute_recall_on_results(results)
            row_results = {"Nombre de sections": m_chunks, "Nombre de centroïdes par section": n_centroids, "k": k,
                           f"Rappel @ {k}": recall,
                           "total_time": time_total,
                           "Nombre de requêtes par seconde": int(len(images) / time_total)}
            results_bench.append(row_results)
            logger.info(row_results)
            results_bench_df = pd.DataFrame(results_bench)
            results_bench_df.to_csv("./bench_results_pq.csv")
    plotting.plot_results(results_bench_df, x=f"Rappel @ {k}", y="Nombre de requêtes par seconde",
                          path_to_save="bench_results_pq_figure_A.png", hue="Nombre de centroïdes par section")
    plotting.plot_results(results_bench_df, x=f"Rappel @ {k}", y="Nombre de requêtes par seconde",
                          path_to_save="bench_results_pq_figure_B.png", hue="Nombre de sections")
