from embedders.embedder import Embedder
from typing import List
from openai import OpenAI
from rag_types.chunk import Chunk

class OpenAIEmbedder(Embedder):
  def __init__(self, openai_api_key: str, dimension: int, model: str = "text-embedding-3-large"):
    self.client = OpenAI(api_key=openai_api_key)
    self.model = model
    self.dimension = dimension

  def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
    if len(chunks) == 0:
      return []
    
    search_texts = [chunk['search_text'] for chunk in chunks]
    response = self.client.embeddings.create(model=self.model, input=search_texts, dimensions=self.dimension)
    vectors = [data.embedding for data in response.data]

    return vectors