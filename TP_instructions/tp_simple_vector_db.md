summary: TP Simple Vector DB
id: tp_simple_vector_db
categories: setup
tags: setup
status: Published
authors: OCTO Technology
Feedback Link: https://github.com/AurelienMassiot/simple_vector_db/issues/new

# TP0 - Implémenter une base de données vecteurs from scratch

## Vue d'ensemble

Duration: 0:01:00

L'objectif de ce TP est créer une simple base de données de vecteurs, comme décrit dans notre article <TODO>.
Bien évidemment, le but ici n'est pas de créer une base de données performante utilisable en production, mais plutôt de l'implémenter pas à pas pour en décortiquer chaque brique.

## Installation des dépendances
Dans un premier temps, nous allons installer le dépendances nécessaires à notre projet. Pour cela, installez `scikit-learn`, `numpy` et `sqlalchemy` dans votre environnement Python. Il y a plusieurs façosn de le faire : Conda, Pipenv, Poetry, etc. Libre à vous d'utiliser la méthode que vous préférez. Ici, nous allons utiliser Poetry en créant un fichier `pyproject.toml` et en y ajoutant les dépendances nécessaires :

```toml
# pyproject.toml
[tool.poetry]
name = "simple-vector-db"
version = "0.1.0"
description = ""
authors = ["Aurelien Massiot <aurelien.massiot@octo.com>", "Philippe Stepniewski <philippe.stepniewski@octo.com>"]
readme = "README.md"
packages = [{include = "simple_vector_db"}]

[tool.poetry.dependencies]
python = "^3.9"
numpy = "^1.25.2"
scikit-learn = "^1.3.0"
sqlalchemy = "^2.0.20"


[tool.poetry.group.dev.dependencies]
pytest = "^7.4.1"
black = "^23.7.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## Création d'une classe abstraite Vector DB
Nous allons créer un package `simple_vector_db` qui contiendra un certain nombre de fichiers python. N'oubliez donc pas d'y créer un fichier `__init__.py`!

Maintenant, nous allons créer un fichier `vector_db.py` et y créer une classe abstraite VectorDB. Cette classe abstraite contiendra les méthodes `insert`, `search` et `retrieve` que nous allons implémenter dans les prochaines parties.

```python
# simple_vector_db/vector_db.py
from abc import ABC, abstractmethod

class VectorDB(ABC):
    @abstractmethod
    def insert(self, vectors):
        pass

    @abstractmethod
    def search(self, query_vector, k):
        pass

    @abstractmethod
    def retrieve(self, key):
        pass
```

Les méthodes `insert` et `retrieve` sont assez explicites : elles permettent d'insérer des vecteurs et d'en retrouver un par sa clé. La méthode `search` prend en entrée un vecteur de requête et un nombre `k` et pour retourner les `k` vecteurs les plus proches de la requête.


## Calcul de distances entre les vecteurs

Pour calculer des distances entre les vecteurs, il existe plusieurs méthodes et nous allons en implémenter deux : la distance euclidienne et la similarité cosinus. Pour cela, commençons par créer un fichier `distances.py`.
Les méthodes `cosine_similarity` et `euclidean_distance` prennent toutes les deux en entrée deux vecteurs et retournent un nombre entre 0 et 1. Plus ce nombre est proche de 1, plus les vecteurs sont similaires.

```python
# simple_vector_db/distances.py
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.linalg.norm(v1 - v2)
```

## Implémentation d'une base de données vecteurs en mémoire
Maintenant que nous avons une classe abstraite VectorDB et des méthodes pour calculer des distances entre vecteurs, nous allons implémenter une base de données vecteurs en mémoire. Pour cela, nous allons créer un fichier `simple_vector_db/vector_db_in_memory.py` et y créer une classe `InMemoryVectorDB` qui hérite de `VectorDB`.  

Ici, nous implémentons les méthodes `insert`, `search` et `retrieve` en utilisant un dictionnaire python pour stocker les vecteurs. La clé de chaque vecteur est son index dans le dictionnaire.  

Pour chercher les vecteurs les plus proches d'une requête, nous calculons la distance entre la requête et tous les vecteurs de la base de données, puis nous trions les résultats par ordre décroissant de distance et nous ne gardons que les `k` premiers. Il est donc possible ici d'utiliser la distance eucliennne ou la similarité cosinus.

```python
#simple_vector_db/vector_db_in_memory.py
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from simple_vector_db.distances import cosine_similarity
from simple_vector_db.vector_db import VectorDB


class VectorDBInMemory(VectorDB):
    def __init__(self):
        self.vectors = defaultdict(np.ndarray)

    def insert(self, vectors: list[np.ndarray]) -> None:
        for i in range(len(vectors)):
            self.vectors[i] = vectors[i]

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        similarities = [
            (key, cosine_similarity(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        # similarities = [(key, euclidean_distance(query_vector, vector)) for key, vector in self.vectors.items()]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def retrieve(self, key: int) -> np.ndarray:
        return self.vectors.get(key, None)
```

## Ajout de logging
Pour pouvoir suivre l'exécution de notre code, nous allons ajouter un peu de logging coloré (copyright à notre cher [Octo Thomas](https://github.com/AnOtterGithubUser) . Pour cela, nous allons créer un fichier `utils/flex_logging.py` et y créer un handler qui permettra d'afficher les logs dans la console. C'est clairement un nice-to-have non indispensable, mais autant faire les choses bien :

```python
# utils/flex_logging.py
import logging
import sys


class CustomFormatter(logging.Formatter):
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s - (%(filename)s:%(lineno)s)"

    FORMATS = {
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(CustomFormatter())
```


## Test de l'implémentation de la base de données vecteurs en mémoire
Maintenant que nous avons implémenté notre première base de données vecteurs, testons la. Nous allons créer un fichier `main_in_memory.py` dans lequel nous allons pouvoir tester notre implémentation.

Pour cela, nous allons créer une base de données vecteurs en mémoire, y insérer 3 vecteurs sous forme de numpy array, en retrouver un par sa clé et chercher les k vecteurs les plus similaires de notre vecteur requête.
N'hésitez pas à changer les valeurs des vecteurs (les vecteurs insérés `vectors_to_insert` au préalable et le vecteur de requête `QUERY_VECTOR`).

Notez l'import de notre librairie de logging et l'utilisation de notre handler `stream_handler` pour afficher les logs dans la console.

A ce stade, l'arborescence de notre projet devrait ressembler à cela :

```texte simple
    .
    └── simple_vector_db
        ├── simple_vector_db
        | ├── __init__.py
        | ├── distances.py
        | ├── vector_db.py
        | ├── vector_db_in_memory.py
        ├── utils
        | ├── flex_logging.py
        poetry.lock
        pyproject.toml
        main_in_memory.py
```

```python
# main_in_memory.py
import logging

import numpy as np

from simple_vector_db.vector_db_in_memory import VectorDBInMemory
from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

vector_db = VectorDBInMemory()
QUERY_VECTOR = np.array([0.1, 0.2, 0.3])
K_SIMILAR_VECTORS = 3


def perform_search_in_memory():
    vectors_to_insert = [
        np.array([10, 20, 30]),
        np.array([-1, -2, -3]),
        np.array([0.3, 0.3, 0.3]),
    ]
    vector_db.insert(vectors_to_insert)

    retrieved_vector = vector_db.retrieve(1)
    logger.info(f"Retrieved vectors: {retrieved_vector}")

    similar_vectors = vector_db.search(query_vector=QUERY_VECTOR, k=K_SIMILAR_VECTORS)
    logger.info(f"Most {K_SIMILAR_VECTORS} Similar vectors: {similar_vectors}")


if __name__ == "__main__":
    perform_search_in_memory()
```

Et tadaaa !  
Nous pouvons voir que les k vecteurs les plus similaires ont été retournés, ordonnés par ordre décroissant de similarité.

![Run de l'implémentation de la base de données vecteurs en mémoire](images/run_main_in_memory.png "Run de l'implémentation de la base de données vecteurs en mémoire")

Nous allons désormais passer aux choses sérieuses et implémenter une base de données vecteurs persistante.

## Ajout d'un décorateur pour mesurer le temps d'exécution
Avant d'implémenter notre base de données persistante, nous allons d'abord créer un petit utilitaire nous permettant de mesurer le temps d'exécution de nos méthodes. Pour cela, nous allons créer un fichier `utils/timer.py` et y créer un décorateur `timer` qui prend en entrée une fonction et retourne une fonction qui mesure le temps d'exécution de la fonction passée en entrée.  
Créez un fichier `utils/timing.py` et implémentez le décorateur `timer` :

```python
import logging
import time
from functools import wraps

from utils.flex_logging import stream_handler

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(
            f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds"
        )
        return result

    return timeit_wrapper
```

## Premiers objets avec SQLite et SQLAlchemy
Nous allons maintenant créer une base de données SQLite avec SQLAlchemy. SQLAlchemy va grandement nous faciliter la tâche en tant qu'ORM. Pour cela, nous allons créer un fichier `simple_vector_db/vector_db_sqlite.py` et y créer une classe `VectorDBSQLite`.  
L'initialisation de cet objet va nous permettre d'instancier la connexion à la base de données SQLite : 

```python
# simple_vector_db/vector_db_sqlite.py

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
Base = declarative_base()

class VectorDBSQLite():
    def __init__(self, db_filename: str):
        self.engine = create_engine(f"sqlite:///{db_filename}", connect_args={'detect_types': sqlite3.PARSE_DECLTYPES})
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
```

Dans ce même fichier, nous allons ajouter une classe Vector qui va nous permettre de représenter un vecteur dans notre base de données. Cette classe va hériter de la classe `Base` de SQLAlchemy et va contenir un attribut `id` qui sera la clé primaire de notre table et un attribut `vector` qui sera le vecteur stocké dans la base de données.  

```python

from sqlalchemy import  Column, Integer
from simple_vector_db.numpy_array_adapter import NumpyArrayAdapter

class Vector(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True)
    data = Column(NumpyArrayAdapter)

    def __init__(self, data):
        self.data = data
```

Nous pouvons voir qu'ici, notre vecteur est de type NumpyArrayAdapter, et qu'il est importé depuis `simple_vector_db.numpy_array_adapter`. Ce fichier n'existe pas ? C'est normal, nous allons le créer tout de suite.  

## Implémentation d'un adaptateur SQLite pour les vecteurs Numpy
Il n'est pas possible de base de stocker des vecteurs Numpy dans une base de données SQLite. Pour cela, nous allons créer un adaptateur qui va nous permettre de convertir nos vecteurs Numpy en un type que SQLite peut stocker. Pour cela, nous allons créer un fichier `simple_vector_db/numpy_array_adapter.py` et y créer une classe `NumpyArrayAdapter` qui va hériter de la classe `types.TypeDecorator` de SQLAlchemy.  

```python
# simple_vector_db/numpy_array_adapter.py

import numpy as np
from sqlalchemy import types


class NumpyArrayAdapter(types.TypeDecorator):
    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        if value is not None:
            return value.tobytes()

    def process_result_value(self, value, dialect):
        if value is not None:
            return np.frombuffer(value, dtype=np.float64)
```

## Création de la méthode d'insertion des vecteurs

Nous allons désormais pouvoir insérer des vecteurs dans notre base de données. Pour cela, nous allons ajouter une méthode `insert` à notre classe `VectorDBSQLite`. Cette méthode va prendre en entrée une liste de vecteurs et va les insérer dans la base de données.  

```python
# simple_vector_db/vector_db_sqlite.py
class VectorDBSQLite():
    ... 
    
    def insert(self, vectors: list[np.ndarray], vector_ids: list[int] = None) -> None:
        vector_objects = [Vector(data=array) for array in vectors]
        self.session.add_all(vector_objects)
        self.session.commit()
```

On peut créer un script `main_sqlite.py` pour tester une insertion. Nous allons créer une base de données SQLite dans un fichier `vector_db.db` et y insérer 1000 vecteurs de dimension (1, 3).  

```python
# main_sqlite.py

import numpy as np
from simple_vector_db.vector_db_sqlite import VectorDBSQLite

DB_FILENAME = "vector_db.db"
vector_db = VectorDBSQLite(db_filename=DB_FILENAME)

N_VECTORS = 1000
VECTORS_SHAPE = (1, 3)

def insert_vectors(n_vectors: int, vectors_shape: tuple[int, int]) -> None:
    vectors_to_insert = [np.random.rand(*vectors_shape) for _ in range(n_vectors)]
    vector_db.insert(vectors_to_insert)

if __name__ == "__main__":
    insert_vectors(N_VECTORS, VECTORS_SHAPE)
```

## Création de la méthode de recherche des vecteurs les plus similaires

Nous allons maintenant implémenter la méthode `search` de notre classe `VectorDBSQLite`. Cette méthode va prendre en entrée un vecteur de requête et un nombre `k` et va retourner les `k` vecteurs les plus similaires à la requête.  

```python
# simple_vector_db/vector_db_sqlite.py
from typing import List, Tuple
from simple_vector_db.distances import cosine_similarity

class VectorDBSQLite():
    ...
    
    def search_without_index(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        vectors = self.session.query(Vector).all()

        similarities = [
            (vector.id, cosine_similarity(query_vector, vector.data))
            for vector in vectors
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:k]

        return top_similarities
```

De même que tout à l'heure avec l'implémentation en mémoire, il est possible de changer la distance utilisée pour la recherche. Ici, nous utilisons la similarité cosinus.

##


