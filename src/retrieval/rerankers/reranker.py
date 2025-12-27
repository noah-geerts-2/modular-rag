from abc import ABC, abstractmethod
from typing import List

from common.rag_types import Chunk

class Reranker(ABC):
    @abstractmethod
    def rerank(self, chunks: List[Chunk], query: str) -> List[Chunk]:
        pass