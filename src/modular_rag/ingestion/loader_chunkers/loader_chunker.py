from abc import ABC, abstractmethod
from typing import List, Set
from modular_rag.common.rag_types import Chunk

class LoaderChunker(ABC):
  @property
  @abstractmethod
  def supported_extensions(self) -> Set[str]:
      pass

  @abstractmethod
  def load_and_chunk(self, path: str) -> List[Chunk]:
    pass