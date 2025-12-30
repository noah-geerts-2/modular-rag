from typing import List
from modular_rag.common.rag_types import Chunk
from .reranker import Reranker
from pinecone import Pinecone

class PineconeReranker(Reranker):
    def __init__(self, pc: Pinecone, finalK: int = 3, model: str = "bge-reranker-v2-m3"):
        self.finalK = finalK
        self.pc = pc
        self.model = model

    def rerank(self, chunks: List[Chunk], ids: List[int], query: str) -> List[Chunk]:
        # Assertion
        if len(chunks) != len(ids):
            raise RuntimeError("The number of chunks must match the number of queries")
        if len(chunks) == 0:
            raise RuntimeError("Input chunks and ids cannot be empty")

        # Rerank search_text against query for each chunk using pinecone
        documents = [{"id": id, "search_text": chunk["search_text"]} for chunk, id in zip(chunks, ids)]
        reranked = self.pc.inference.rerank(
            model=self.model,
            query=query,
            documents=documents, 
            top_n=self.finalK,
            return_documents=True,
            rank_fields=["search_text"]
        )

        # Map each id to its chunk
        chunk_map = {}
        for chunk, id in zip(chunks, ids):
            chunk_map[id] = chunk

        # Retrieve the correct chunks from this map based on the reranked ids
        out = [chunk_map[result.document.id] for result in reranked.data]
        return out