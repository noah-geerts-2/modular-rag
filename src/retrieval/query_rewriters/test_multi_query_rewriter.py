import unittest
from unittest.mock import MagicMock

from retrieval.query_rewriters.multi_query_rewriter import MultiQueryRewriter


class TestMultiQueryRewriter(unittest.TestCase):
    def test_rewrite_calls_llm_with_correct_args(self):
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = "q1|--|q2|--|q3"

        rewriter = MultiQueryRewriter(mock_llm, n=3)
        out = rewriter.rewrite_query("find cats")

        mock_llm.create_chat_completion.assert_called_once()
        args, kwargs = mock_llm.create_chat_completion.call_args

        # ensure original query was passed inside the user prompt
        self.assertIn("find cats", args[0])

        # ensure system_message was passed as a keyword argument
        self.assertIn("system_message", kwargs)

        # ensure output parsing works
        self.assertEqual(out, ["q1", "q2", "q3"])


if __name__ == "__main__":
    unittest.main()
