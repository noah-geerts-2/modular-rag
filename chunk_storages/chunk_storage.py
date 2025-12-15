from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class ChunkStorage(ABC):
  @abstractmethod
  # Stores a list of langchain documents by id in some form of persistent storage, returning the id's as a list
  def store_chunks(self, chunks: List[Document]) -> List[int]:
    pass

  @abstractmethod
  # Returns a list of langchain documents corresponding to the provided id'set
  def retrieve_chunks(self, ids: List[int]) -> List[Document]:
    pass