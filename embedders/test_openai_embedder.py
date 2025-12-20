from typing import List
import unittest
from embedders.openai_embedder import OpenAIEmbedder
import dotenv
import os

from rag_types.chunk import Chunk

dotenv.load_dotenv()

class TestOpenAIEmbedder(unittest.TestCase):
  def test_embedding_3_strings(self):
    # Arrange
    toEmbed: List[Chunk] = ["a", "b", "c"]
    embedder = OpenAIEmbedder(os.environ['OPENAI_API_KEY'])

    # Act & Assert
    vectors = embedder.embed_strings(toEmbed)
    self.assertEqual(len(vectors), 3)
    self.assertEqual(len(vectors[0]), 3072)

  def test_embedding_0_strings(self):
    # Arrange
    toEmbed = []
    embedder = OpenAIEmbedder(os.environ['OPENAI_API_KEY'])

    # Act & Assert
    vectors = embedder.embed_strings(toEmbed)
    self.assertEqual(len(vectors), 0)

if __name__ == "__main__":
  unittest.main()