from typing import List
import unittest
from .openai_embedder import OpenAIEmbedder
import dotenv
import os
from openai import OpenAI

from common.rag_types import Chunk

dotenv.load_dotenv()

class TestIntegrationOpenAIEmbedder(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.embedder = OpenAIEmbedder(OpenAI(api_key=os.environ['OPENAI_API_KEY']))

  ####
  # These are integration tests that actually call OpenAI's API
  ####

  def test_embedding_3_strings(self):
    # Arrange
    toEmbed: List[Chunk] = ["a", "b", "c"]

    # Act & Assert
    vectors = self.embedder.embed_strings(toEmbed)
    self.assertEqual(len(vectors), 3)
    self.assertEqual(len(vectors[0]), 3072)

  def test_embedding_0_strings(self):
    # Arrange
    toEmbed = []

    # Act & Assert
    vectors = self.embedder.embed_strings(toEmbed)
    self.assertEqual(len(vectors), 0)

if __name__ == "__main__":
  unittest.main()