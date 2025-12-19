from abc import ABC, abstractmethod
from typing import List, Set

class QueryRewriter(ABC):
  @abstractmethod
  def rewrite(self, query: str) -> List[str]:
    pass