from embedders.embedder import Embedder
from typing import List
from openai import OpenAI
from rag_types.chunk import Chunk

class OpenAIEmbedder(Embedder):
  def __init__(self, openai_api_key: str, dimension: int = 3072, model: str = "text-embedding-3-large"):
    self.client = OpenAI(api_key=openai_api_key)
    self.model = model
    self.dimension = dimension

  def embed_strings(self, strings: List[str]) -> List[List[float]]:
    if len(strings) == 0:
      return []

    response = self.client.embeddings.create(model=self.model, input=strings, dimensions=self.dimension)
    vectors = [data.embedding for data in response.data]

    return vectors