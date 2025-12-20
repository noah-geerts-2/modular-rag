import unittest
from unittest.mock import MagicMock, patch, call

from retrievers.semantic_retriever import SemanticRetriever


class TestSemanticRetriever(unittest.TestCase):
  def test_single_query_returns_top_k_without_rrf(self):
    embedder = MagicMock()
    vector_db = MagicMock()
    embedder.embed_strings.return_value = ["vec1"]
    vector_db.semantic_search.return_value = [
      {"search_text": "a"},
      {"search_text": "b"},
      {"search_text": "c"},
    ]

    retriever = SemanticRetriever(vector_db, embedder, perQueryK=5, finalK=2)

    with patch('retrievers.semantic_retriever.rrf') as mock_rrf:
      result = retriever.retrieve_chunks(["q1"])

    mock_rrf.assert_not_called()
    embedder.embed_strings.assert_called_once_with(["q1"])
    vector_db.semantic_search.assert_called_once_with("vec1", 5)
    self.assertEqual(result, [{"search_text": "a"}, {"search_text": "b"}])

  def test_multiple_queries_uses_rrf_and_combines_subresults(self):
    embedder = MagicMock()
    vector_db = MagicMock()
    embedder.embed_strings.return_value = ["v1", "v2"]
    result_q1 = [{"search_text": "a1"}, {"search_text": "a2"}]
    result_q2 = [{"search_text": "b1"}]
    vector_db.semantic_search.side_effect = [result_q1, result_q2]

    retriever = SemanticRetriever(vector_db, embedder, perQueryK=4, finalK=3)
    rrf_result = [{"search_text": "combined"}]

    with patch('retrievers.semantic_retriever.rrf', return_value=rrf_result) as mock_rrf:
      result = retriever.retrieve_chunks(["q1", "q2"])

    embedder.embed_strings.assert_called_once_with(["q1", "q2"])
    vector_db.semantic_search.assert_has_calls([call("v1", 4), call("v2", 4)])
    mock_rrf.assert_called_once_with([result_q1, result_q2], 3)
    self.assertEqual(result, rrf_result)

  def test_multiple_queries_still_calls_rrf_when_some_results_empty(self):
    embedder = MagicMock()
    vector_db = MagicMock()
    embedder.embed_strings.return_value = ["v1", "v2", "v3"]
    result_q1 = []
    result_q2 = [{"search_text": "b"}]
    result_q3 = []
    vector_db.semantic_search.side_effect = [result_q1, result_q2, result_q3]

    retriever = SemanticRetriever(vector_db, embedder, perQueryK=1, finalK=2)
    rrf_result = [{"search_text": "b"}]

    with patch('retrievers.semantic_retriever.rrf', return_value=rrf_result) as mock_rrf:
      result = retriever.retrieve_chunks(["q1", "q2", "q3"])

    mock_rrf.assert_called_once_with([result_q1, result_q2, result_q3], 2)
    self.assertEqual(result, rrf_result)


if __name__ == '__main__':
  unittest.main()
