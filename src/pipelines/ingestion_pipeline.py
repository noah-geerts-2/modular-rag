from typing import List
from common.rag_types.chunk import Chunk
from ingestion.chunk_dbs import ChunkDB
from common.embedders import Embedder
from ingestion.loader_chunkers import LoaderChunker
from ingestion.vector_dbs import VectorDB


class IngestionPipeline:
  def __init__(self, loaderChunker: LoaderChunker, ChunkDB: ChunkDB, embedder: Embedder, VectorDB: VectorDB):
    self.loaderChunker = loaderChunker
    self.ChunkDB = ChunkDB
    self.embedder = embedder
    self.VectorDB = VectorDB

  # Ingests all files at a given path
  def ingest(self, path: str):
    chunks: List[Chunk] = self.loaderChunker.load_and_chunk(path)
    ids = self.ChunkDB.store_chunks(chunks)
    vectors = self.embedder.embed_strings(chunk['search_text'] for chunk in chunks)
    self.VectorDB.store_embeddings(ids, vectors)
