# Ingestion Pipeline
from pipelines.ingestion_pipeline import IngestionPipeline
from ingestion.loader_chunkers import MultiModalLoaderChunker
from ingestion.chunk_dbs import SQLiteChunkDB
from common.embedders import OpenAIEmbedder
from ingestion.vector_dbs import PineconeVectorDB
from common.llms import ChatGPT
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pc_api_key = os.getenv("PINECONE_API_KEY")
if openai_api_key is None or pc_api_key is None:
    raise ValueError("API keys for OpenAI and Pinecone must be set in environment variables.")

# Create instances of each component
client = OpenAI(api_key=openai_api_key)
llm = ChatGPT(client)
loader_chunker = MultiModalLoaderChunker(llm)

chunk_db = SQLiteChunkDB("test.db", "chunks")

dim = 3072
embedder = OpenAIEmbedder(client, dimension=dim)

pc = Pinecone(api_key=pc_api_key)
vector_db = PineconeVectorDB(pc, "test-index", dimension=dim)

# Create the ingestion pipeline
ingestion_pipeline = IngestionPipeline(loader_chunker, chunk_db, embedder, vector_db)

# Ingest data
ingestion_pipeline.ingest("../documents/")