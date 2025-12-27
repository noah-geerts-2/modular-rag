from loader_chunkers.multimodal_loader_chunker import MultiModalLoaderChunker
from chunk_storages.sqlite_chunk_storage import SQLiteChunkDB
from embedders.openai_embedder import OpenAIEmbedder
from vector_dbs.pinecone_vector_store import PineconeVectorDB
from query_rewriters.multi_query_rewriter import MultiQueryRewriter

from src.pipelines.ingestion_pipeline import IngestionPipeline

import dotenv
import os

dotenv.load_dotenv()

# Extract environment variables
openai_api_key = os.environ['OPENAI_API_KEY']
db_name = os.environ['DB_NAME']
table_name = os.environ['TABLE_NAME']
dimension = int(os.environ['DIMENSION'])
pinecone_api_key = os.environ['PINECONE_API_KEY']
index_name = os.environ['INDEX_NAME']

# print("INITIALIZING PIPELINE\n")

# ingestionPipeline = IngestionPipeline(
#   MultiModalLoaderChunker(openai_api_key),
#   SQLiteChunkDB(db_name, table_name),
#   OpenAIEmbedder(openai_api_key, dimension),
#   PineconeVectorDB(pinecone_api_key, index_name, dimension)
# )

# print("INGESTING:\n\n")

# ingestionPipeline.ingest("./documents")

rewriter = MultiQueryRewriter(openai_api_key, n=3)
print(rewriter.rewrite_query("What's the deadliest heart condition?"))