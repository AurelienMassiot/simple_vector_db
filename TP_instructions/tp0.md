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

## Création d'une classe abstraite Vector DB
Dans un premier temps, nous allons créer un fichier `vector_db.py` et y créer une classe abstraite VectorDB. Cette classe abstraite contiendra les méthodes `insert`, `search` `retrieve` que nous allons implémenter dans les prochaines parties.

```python
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