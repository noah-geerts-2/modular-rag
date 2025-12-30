from abc import ABC, abstractmethod
from typing import List
from modular_rag.common.rag_types import Candidate

class VectorDB(ABC):
  @abstractmethod
  def store_embeddings(self, ids: List[int], vectors: List[List[float]]):
    pass

  @abstractmethod
  def semantic_search(self, query: List[float], k: int) -> List[Candidate]:
    pass