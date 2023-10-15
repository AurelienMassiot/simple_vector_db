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
    images = load_fashion_mnist(n_sample=5000)
    logger.info("Loaded Dataset")
    logger.info("Initiated Vector DB")
    results_bench = []
    for nb_clusters in range(2, 256):
        vector_db = prepare_sql_db()
        k = 50
        ds = Dataset(images, k=k)
        eval = VectorDBEvaluator(vector_db, ds)
        eval.set_kmeans_index(nb_clusters)
        logger.info("Created evaluation Dataset")
        time_start = time.time()
        results = eval.query_with_all_vectors_kmeans_index(k=k)
        time_end = time.time()
        time_total = time_end - time_start
        logger.info("Queried vector db with all data in" + str(time_total))
        recall = eval.compute_recall_on_results(results)
        results_bench.append({"nb_clusters": nb_clusters, "k": k, "recall": recall, "total_time": time_total,
                        "nb_requests_per_second": int(len(images) / time_total)})
        logger.info("Recall is " + str(recall))
        results_bench_df = pd.DataFrame(results_bench)
        results_bench_df.to_csv("figures/bench_results_kmeans.csv")
