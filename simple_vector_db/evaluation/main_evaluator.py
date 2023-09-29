import random
import shutil
from simple_vector_db.evaluation.evaluator import Dataset, VectorDBEvaluator
from simple_vector_db.vector_db_sqlite import VectorDBSQLite
from sklearn.datasets import load_digits
import logging
from utils.flex_logging import stream_handler
<<<<<<< HEAD
import dataget
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)
=======

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
>>>>>>> 46089d5 (feat: finished evaluator computing the recall between ANN and KNN at a given k. For now, cant reach 100% recall because of ties in the knn using cosine sim)


def load_mnist_data():
    digits = load_digits().data
    random.shuffle(digits)
    #digits = digits[0:100]
    return digits

<<<<<<< HEAD
def load_fashion_mnist(n_sample = 1000):
    X_train, y_train, X_test, y_test = dataget.image.fashion_mnist().get()
    X = np.vstack([X_train,X_test])
    images = []
    for image in X:
        images.append(image.flatten())
    random.shuffle(images)
    images = images[0:n_sample]
    return images

=======
>>>>>>> 46089d5 (feat: finished evaluator computing the recall between ANN and KNN at a given k. For now, cant reach 100% recall because of ties in the knn using cosine sim)
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
<<<<<<< HEAD
    vector_db.set_metric("euclidean")
    return vector_db

if __name__ == "__main__":
    images = load_fashion_mnist(n_sample=100)
    logger.info("Loaded Dataset")
    vector_db = prepare_sql_db()
    logger.info("Initiated Vector DB")
    k = 50
    ds = Dataset(images, k=k)
    eval = VectorDBEvaluator(vector_db, ds)
    logger.info("Created evaluation Dataset")
    results = eval.query_with_all_vectors(k=k)
    logger.info("Queried vector db with all data")
=======
    return vector_db

if __name__ == "__main__":
    digits = load_mnist_data()
    vector_db = prepare_sql_db()
    k = 10
    ds = Dataset(digits, k=k)
    eval = VectorDBEvaluator(vector_db, ds)
    results = eval.query_with_all_vectors(k=k)
>>>>>>> 46089d5 (feat: finished evaluator computing the recall between ANN and KNN at a given k. For now, cant reach 100% recall because of ties in the knn using cosine sim)
    recall = eval.compute_recall_on_results(results)
    logger.info(f"Computed RECALL @ {k} is {str(recall)}")
    remove_sqlite_file()
