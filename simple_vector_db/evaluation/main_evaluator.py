import random
import shutil

import pandas as pd

from simple_vector_db.evaluation.evaluator import Dataset, VectorDBEvaluator
from simple_vector_db.vector_db_sqlite import VectorDBSQLite
from sklearn.datasets import load_digits
import logging
from utils.flex_logging import stream_handler
import dataget
import numpy as np
import time

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


def prepare_sql_db():
    db_filename = "test_db.db"
    remove_sqlite_file()
    vector_db = VectorDBSQLite(db_filename)
    vector_db.set_metric("euclidean")
    return vector_db


if __name__ == "__main__":
    remove_sqlite_file()
    images = load_fashion_mnist(n_sample=1000)
    logger.info("Loaded Dataset")
    logger.info("Initiated Vector DB")
    results_bench = []
    for nb_clusters in range(2, 128):
        for n_probes in range(1, 5):
            vector_db = prepare_sql_db()
            k = 10
            ds = Dataset(images, k=k)
            eval = VectorDBEvaluator(vector_db, ds)
            eval.set_kmeans_index(nb_clusters)
            logger.info("Created evaluation Dataset")
            time_start = time.time()
            results = eval.query_with_all_vectors_kmeans_index(k=k, n_probes=n_probes)
            time_end = time.time()
            time_total = time_end - time_start
            recall = eval.compute_recall_on_results(results)
            row_results = {"Nombre de clusters": nb_clusters, "Valeur de n_probes": n_probes, "k": k, f"Rappel @ {k}": recall, "total_time": time_total,
                 "Nombre de requêtes par seconde": int(len(images) / time_total)}
            results_bench.append(row_results)
            logger.info(row_results)
            results_bench_df = pd.DataFrame(results_bench)
            results_bench_df.to_csv("figures/bench_results_kmeans.csv")
