from openai import OpenAI
from pinecone import Pinecone
from modular_rag.common.embedders.openai_embedder import OpenAIEmbedder
from modular_rag.ingestion.chunk_dbs import SQLiteChunkDB
from modular_rag.ingestion.vector_dbs.pinecone_vector_db import PineconeVectorDB
from modular_rag.pipelines import RetrievalPipeline
from modular_rag.retrieval.retrievers import SemanticRetriever

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pc_api_key = os.getenv("PINECONE_API_KEY")
if openai_api_key is None or pc_api_key is None:
    raise ValueError("API keys for OpenAI and Pinecone must be set in environment variables.")

# Create instances of each component
dim = 3072

client = OpenAI(api_key=openai_api_key)
embedder = OpenAIEmbedder(client, dimension=dim) # Embedder using OpenAI

pc = Pinecone(api_key=pc_api_key)
vector_db = PineconeVectorDB(pc, "test-index", dimension=dim) # VectorDB using Pinecone

chunk_db = SQLiteChunkDB("test.db", "chunks") # ChunkDB
retriever = SemanticRetriever(vector_db, embedder, semanticK=3, finalK=3) # Retriever using VectorDB and Embedder

# Pipeline
retrieval_pipeline = RetrievalPipeline(retriever, chunk_db)
results = retrieval_pipeline.retrieve("How will LLM's affect the outcome of our society?")
for result in results:
    print(result)