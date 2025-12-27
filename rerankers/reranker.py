from abc import ABC, abstractmethod
from typing import List, Set

from rag_types.chunk import Chunk

class Reranker(ABC):
    @abstractmethod
    def rerank(self, chunks: List[Chunk], query: str) -> List[Chunk]:
        pass