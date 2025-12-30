from typing import List
from modular_rag.common.rag_types.candidate import Candidate
from modular_rag.ingestion.chunk_dbs import ChunkDB
from modular_rag.retrieval.query_rewriters import QueryRewriter
from modular_rag.retrieval.rerankers import Reranker
from modular_rag.retrieval.retrievers import Retriever


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        chunkDB: ChunkDB,
        queryRewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None
    ):
        self.queryRewriter = queryRewriter
        self.retriever = retriever
        self.chunkDB = chunkDB
        self.reranker = reranker

    def retrieve(self, query: str):
        # Rewrite the query if a rewriter is provided
        queries = [query]
        if self.queryRewriter:
            queries = self.queryRewriter.rewrite_query(query)

        # Retrieve candidates using retriever
        candidate: List[Candidate] = self.retriever.retrieve_candidates(queries)

        # Extract candidate IDs and retrieve the actual chunks
        ids = [candidate["id"] for candidate in candidate]
        chunks = self.chunkDB.retrieve_chunks(ids)

        # Rerank the chunks if a reranker is provided
        if self.reranker:
            chunks = self.reranker.rerank(chunks, ids, query)

        return chunks