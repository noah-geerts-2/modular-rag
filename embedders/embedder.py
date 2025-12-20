from abc import ABC, abstractmethod
from typing import List
from rag_types.chunk import Chunk

class Embedder(ABC):
  @abstractmethod
  def embed_strings(self, strings: List[str]) -> List[List[float]]:
    pass