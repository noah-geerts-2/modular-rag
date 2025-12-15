from chunk_storage import ChunkStorage
from typing import List, Optional
from langchain_core.documents import Document
import sqlite3
import json

class SQLiteChunkStorage(ChunkStorage):

  def __init__(self, db_name: str, table_name: str):
    if not db_name:
      raise RuntimeError("SQLiteChunkStorage requires a db_name.")
    if not table_name:
      raise RuntimeError("SQLiteChunkStorage requires a table_name.")

    self.db_name = db_name
    self.table_name = table_name

    self.conn = sqlite3.connect(self.db_name)
    self.cur = self.conn.cursor()

    existing_table = self.cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (self.table_name,)
    ).fetchone()

    if existing_table is None:
      self.cur.execute(
        f"CREATE TABLE {self.table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_json TEXT)"
      )
      self.conn.commit()

  def store_chunks(self, chunks: List[Document]) -> List[int]:
    ids = []
    for chunk in chunks:
      chunk_dictionary = {"page_content": chunk.page_content, "metadata": chunk.metadata}
      chunk_json = json.dumps(chunk_dictionary)
      self.cur.execute(f"INSERT INTO {self.table_name} (chunk_json) VALUES (?) RETURNING id", (chunk_json,))
      ids.append(self.cur.fetchone()[0])
    self.conn.commit()
    return ids
  
  def retrieve_chunks(self, ids: List[int]) -> List[Document]:
    if not ids:
      return []
    id_placeholders = ",".join("?" for _ in ids)
    self.cur.execute(f"SELECT chunk_json FROM {self.table_name} WHERE id IN ({id_placeholders})", ids)
    rows = self.cur.fetchall()
    dictionaries = [json.loads(row[0]) for row in rows]
    documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in dictionaries]
    return documents