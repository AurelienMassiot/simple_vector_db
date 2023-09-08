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
