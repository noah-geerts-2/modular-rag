from abc import ABC, abstractmethod
from typing import List, Set

class QueryRewriter(ABC):
  @abstractmethod
  def rewrite_query(self, query: str) -> List[str]:
    pass