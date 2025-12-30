from abc import ABC, abstractmethod
from typing import List
from modular_rag.common.rag_types import Chunk

class ChunkDB(ABC):
  @abstractmethod
  # Stores a list of chunks by id in some form of persistent storage, returning the id's as a list
  def store_chunks(self, chunks: List[Chunk]) -> List[int]:
    pass

  @abstractmethod
  # Returns a list of langchain chunks corresponding to the provided id's
  def retrieve_chunks(self, ids: List[int]) -> List[Chunk]:
    pass