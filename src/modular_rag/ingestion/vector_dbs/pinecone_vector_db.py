from typing import List
from .vector_db import VectorDB
from pinecone import Pinecone, QueryResponse, ServerlessSpec, Vector
from modular_rag.common.rag_types import Candidate

class PineconeVectorDB(VectorDB):
  def __init__(self, pc: Pinecone, index_name: str, dimension: int, cloud: str = "aws", region: str = "us-east-1"):
    self.pc = pc
    self.index_name = index_name
    self.dimension = dimension

    # Create the index if it doesn't exist
    if self.index_name not in [idx["name"] for idx in self.pc.list_indexes()]:
      self.pc.create_index(
        name=self.index_name,
        dimension=self.dimension,
        spec=ServerlessSpec(cloud=cloud, region=region),
      )

    # Get index handle
    self.index = self.pc.Index(name=self.index_name)

  def store_embeddings(self, ids: List[int], vectors: List[List[float]]):
    # Prepare items: id must be str
    
    upserts: List[Vector] = [
      Vector(str(i), v)
      for i, v in zip(ids, vectors)
    ]
    self.index.upsert(vectors=upserts)

  def semantic_search(self, query: List[float], k: int) -> List[Candidate]:
    if k < 1:
      raise RuntimeError("K must be at least 1 for semantic search")
    if len(query) != self.dimension:
      raise RuntimeError(f"The dimension of the query vector must be the same as the data vectors ({self.dimension})")
    
    res = self.index.query(vector=query, top_k=k, include_values=False)
    if not isinstance(res, QueryResponse):
      raise RuntimeError("Pinecone's index.query function returned an async reponse instead of a QueryResponse entity")
    candidates: List[Candidate] = [{"id": candidate["id"], "score": candidate['score']} for candidate in res.matches]
    return candidates