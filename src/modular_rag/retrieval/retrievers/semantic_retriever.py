from typing import List
from modular_rag.common.rag_types import Candidate
from .retriever import Retriever, rrf
from modular_rag.ingestion.vector_dbs import VectorDB
from modular_rag.common.embedders import Embedder
from concurrent.futures import ThreadPoolExecutor
import itertools

class SemanticRetriever(Retriever):
  def __init__(self, vectorDb: VectorDB, embedder: Embedder, semanticK: int = 10, finalK: int = 3):
    self.vectorDb = vectorDb
    self.embedder = embedder
    self.perQueryK = semanticK
    self.finalK = finalK

  def retrieve_candidates(self, queries: List[str]) -> List[Candidate]:
    # Check for no queries (makes no sense.. we can't retrieve for nothing)
    N = len(queries)
    if N == 0:
      raise RuntimeError("No queries were provided to the SemanticRetriever's retrieve_candidates method")

    # Embed each query for vector search
    queryVectors = self.embedder.embed_strings(queries)

    # For each query, do retrieval (in parallel to reduce number of network RTT's)
    with ThreadPoolExecutor(max_workers=N) as ex:
      subresults = list(ex.map(self.vectorDb.semantic_search, queryVectors, itertools.repeat(self.perQueryK)))

    # Perform RRF if there is more than one subresult
    if len(subresults) == 1:
      return subresults[0][:self.finalK]
    else:
      return rrf(subresults, self.finalK)