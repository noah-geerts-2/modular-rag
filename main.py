from typing import List
from loader_chunkers.multimodal_loader_chunker import MultiModalLoaderChunker
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

chunkerLoader = MultiModalLoaderChunker()

chunks: List[Document] = chunkerLoader.load_and_chunk("./documents")

with open("chunks.txt", "w", encoding="utf-8") as f:
  for i, chunk in enumerate(chunks):
    f.write(f"--- Chunk {i} ---\n")
    f.write(f"Content: {chunk.page_content}\n")
    f.write(f"Metadata: {chunk.metadata}\n\n")