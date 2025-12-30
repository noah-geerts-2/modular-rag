import unittest
from common.embedders.openai_embedder import OpenAIEmbedder
from unittest.mock import Mock, patch, MagicMock

class TestOpenAIEmbedder(unittest.TestCase):
    def test_embed_strings_calls_openai_with_correct_params(self):
        # Arrange
        openai = MagicMock()
        embedder = OpenAIEmbedder(openai=openai, dimension=500, model="best-model")
        to_embed = ["test string 1", "test string 2"]

        # Act
        embedder.embed_strings(to_embed)

        # Assert
        openai.embeddings.create.assert_called_once_with(model="best-model", input=to_embed, dimensions=500)
