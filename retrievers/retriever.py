from abc import ABC, abstractmethod
from typing import List
from rag_types.chunk import Chunk

def rrf(subresults: List[List[Chunk]], finalK: int, c: int = 60) -> List[Chunk]:
  if len(subresults) <= 1:
    raise RuntimeError("We need at least 2 subresults to perform RRF")
  
  scores = {}

  # Iterate through every chunk in every subresult
  for subresult in subresults:
    for rank, chunk in enumerate(subresult):
      # Compute RRF score addition
      additional_score = 1 / (c + rank + 1)

      if chunk['search_text'] not in scores:
        scores[chunk['search_text']] = [additional_score, chunk]
      else:
        scores[chunk['search_text']][0] += additional_score
  
  # Take the chunks with the finalK highest RRF scores
  scored = [(tup[0], tup[1]) for tup in scores.values()]
  scored.sort(key = lambda x: x[0], reverse = True)
  return [chunk for _, chunk in scored[:finalK]]

class Retriever(ABC):
  @abstractmethod
  def retrieve_chunks(self, queries: List[str]) -> List[Chunk]:
    pass