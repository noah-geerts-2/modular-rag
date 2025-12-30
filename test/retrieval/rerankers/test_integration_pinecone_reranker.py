import unittest
from dotenv import load_dotenv
import os

from pinecone import Pinecone
from modular_rag.retrieval.rerankers import PineconeReranker

load_dotenv()


class TestPineconeReranker(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.pc = Pinecone(os.environ.get('PINECONE_API_KEY'))
		cls.reranker = PineconeReranker(cls.pc)

	def test_length_mismatch_raises(self):
		chunks = [{"search_text": "a"}, {"search_text": "b"}]
		ids = [0]
		with self.assertRaises(RuntimeError):
			self.reranker.rerank(chunks, ids, "query")

	def test_empty_inputs_raises(self):
		chunks = []
		ids = []
		with self.assertRaises(RuntimeError):
			self.reranker.rerank(chunks, ids, "any")
			
	def test_more_chunks_than_finalK_truncates_and_orders(self):
		# design chunks so some have higher overlap with query
		chunks = [
			{"search_text": "red person"},
			{"search_text": "red people"},
			{"search_text": "red"},
			{"search_text": "blue car"},
			{"search_text": "green tree"},
		]
		ids = [0, 1, 2, 3, 4]

		out = self.reranker.rerank(chunks, ids, "red people")

		expected = [chunks[1], chunks[0], chunks[2]]
		self.assertEqual(out, expected)

	def test_same_or_fewer_chunks_than_finalK_reorders_all(self):
		chunks = [
			{"search_text": "red red"},
			{"search_text": "red person"},
		]
		ids = [0, 1]

		out = self.reranker.rerank(chunks, ids, "red people")

		expected = [chunks[1], chunks[0]]
		self.assertEqual(out, expected)


if __name__ == "__main__":
	unittest.main()
