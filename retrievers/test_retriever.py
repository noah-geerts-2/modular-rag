import unittest

from retrievers.retriever import rrf


class TestRRF(unittest.TestCase):
  def test_rrf_ranks_in_expected_order(self):
    subresults = [
      [{'search_text': 'A'}, {'search_text': 'B'}, {'search_text': 'C'}],
      [{'search_text': 'B'}, {'search_text': 'C'}, {'search_text': 'A'}],
      [{'search_text': 'B'}, {'search_text': 'D'}, {'search_text': 'A'}],
    ]

    ranked = rrf(subresults, finalK = 3, c = 0)
    self.assertEqual([chunk['search_text'] for chunk in ranked], ['B', 'A', 'C'])

  def test_rrf_requires_multiple_subresults(self):
    with self.assertRaises(RuntimeError):
      rrf([[]], finalK = 2)

    with self.assertRaises(RuntimeError):
      rrf([], finalK = 2)

  def test_rrf_handles_empty_lists_and_truncates(self):
    subresults = [
      [],
      [{'search_text': 'only'}, {'search_text': 'second'}],
    ]

    ranked = rrf(subresults, finalK = 5, c = 0)
    self.assertEqual([chunk['search_text'] for chunk in ranked], ['only', 'second'])

  def test_rrf_respects_finalk_and_aggregates_scores(self):
    subresults = [
      [{'search_text': 'X'}, {'search_text': 'Y'}],
      [{'search_text': 'Y'}, {'search_text': 'Z'}],
    ]

    ranked = rrf(subresults, finalK = 2, c = 0)
    self.assertEqual([chunk['search_text'] for chunk in ranked], ['Y', 'X'])
    self.assertEqual(len(ranked), 2)


if __name__ == '__main__':
  unittest.main()
