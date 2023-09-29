import random
import shutil
from simple_vector_db.evaluation.evaluator import Dataset, VectorDBEvaluator
from simple_vector_db.vector_db_sqlite import VectorDBSQLite
from sklearn.datasets import load_digits
import logging
from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def load_mnist_data():
    digits = load_digits().data
    random.shuffle(digits)
    #digits = digits[0:100]
    return digits

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
    return vector_db

if __name__ == "__main__":
    digits = load_mnist_data()
    vector_db = prepare_sql_db()
    k = 10
    ds = Dataset(digits, k=k)
    eval = VectorDBEvaluator(vector_db, ds)
    results = eval.query_with_all_vectors(k=k)
    recall = eval.compute_recall_on_results(results)
    logger.info(f"Computed RECALL @ {k} is {str(recall)}")
    remove_sqlite_file()
