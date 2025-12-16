from chunk_storages.chunk_storage import ChunkStorage
from embedders.embedder import Embedder
from loader_chunkers.loader_chunker import LoaderChunker
from vector_stores.vector_store import VectorStore


class IngestionPipeline:
  def __init__(self, loaderChunker: LoaderChunker, chunkStorage: ChunkStorage, embedder: Embedder, vectorStore: VectorStore):
    self.loaderChunker = loaderChunker
    self.chunkStorage = chunkStorage
    self.embedder = embedder
    self.vectorStore = vectorStore

  # Ingests all files at a given path
  def ingest(self, path: str):
    chunks = self.loaderChunker.load_and_chunk(path)
    ids = self.chunkStorage.store_chunks(chunks)
    vectors = self.embedder.embed_chunks(chunks)
    self.vectorStore.store_embeddings(ids, vectors)
