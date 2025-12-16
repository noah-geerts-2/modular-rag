from abc import ABC, abstractmethod
from typing import List
from rag_types.vector import SemanticCandidate

class VectorStore(ABC):
  @abstractmethod
  def store_embeddings(self, ids: List[int], vectors: List[List[float]]):
    pass

  @abstractmethod
  def semantic_search(self, query: List[float], k: int) -> List[SemanticCandidate]:
    pass