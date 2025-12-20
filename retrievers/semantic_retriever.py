from typing import List
from retrievers.retriever import Retriever, rrf
from rag_types.chunk import Chunk
from vector_stores.vector_store import VectorStore
from embedders.embedder import Embedder
from concurrent.futures import ThreadPoolExecutor
import itertools

class SemanticRetriever(Retriever):
  def __init__(self, vectorDb: VectorStore, embedder: Embedder, perQueryK: int = 10, finalK: int = 3):
    self.vectorDb = vectorDb
    self.embedder = embedder
    self.perQueryK = perQueryK
    self.finalK = finalK

  def retrieve_chunks(self, queries: List[str]) -> List[Chunk]:
    # Embed each query for vector search
    queryVectors = self.embedder.embed_strings(queries)

    # For each query, do retrieval (in parallel to reduce number of network RTT's)
    N = len(queries)
    with ThreadPoolExecutor(max_workers=N) as ex:
      subresults = list(ex.map(self.vectorDb.semantic_search, queryVectors, itertools.repeat(self.perQueryK)))

    # Perform RRF if there is more than one subresult
    if len(subresults) == 1:
      return subresults[0][:self.finalK]
    else:
      return rrf(subresults, self.finalK)