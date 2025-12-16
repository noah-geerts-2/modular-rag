import unittest
import os
import dotenv
from vector_stores.pinecone_vector_store import PineconeVectorStore
from pinecone import Pinecone

dotenv.load_dotenv()

# Test constants
TEST_INDEX_NAME = "test"
TEST_DIMENSION = 10

class TestPineconeVectorStore(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Initialize Pinecone client
    cls.api_key = os.environ['PINECONE_API_KEY']
    cls.pc = Pinecone(api_key=cls.api_key)
    
    # Check if index already exists - if so, raise error
    existing_indexes = [idx.name for idx in cls.pc.list_indexes()]
    if TEST_INDEX_NAME in existing_indexes:
      raise RuntimeError(f"Index '{TEST_INDEX_NAME}' already exists. Please delete it before running tests.")
    
    # Create the vector store (which will create the index)
    cls.vector_store = PineconeVectorStore(
      pinecone_api_key=cls.api_key,
      index_name=TEST_INDEX_NAME,
      dimension=TEST_DIMENSION
    )

  # @classmethod
  # def tearDownClass(cls):
  #   # Delete the test index
  #   cls.pc.delete_index(TEST_INDEX_NAME)

  def test_cluster_query_returns_nearest_neighbors(self):
    # Arrange: Create 10 vectors - 7 in a cluster around [1,1,1,...], 3 far away
    vectors = []
    ids = []
    
    # Create 7 vectors close to each other IN ANGLE indexes 0-6
    for i in range(7):
      vector = [0.01 + (i * 0.001)] + [1.0] * (TEST_DIMENSION - 1)  # All point to about .01 in the first dimension
      vectors.append(vector)
      ids.append(i)
    
    # Create 3 vectors far away with indexes 7-9
    for i in range(7, 10):
      vector = [0.9 + ((i-7) * 0.001)] + [1.0] * (TEST_DIMENSION - 1) # All point to about .9 in the first dimension
      vectors.append(vector)
      ids.append(i)
    
    # Store all vectors
    self.vector_store.store_embeddings(ids, vectors)
    
    # Act: Query with a vector close to the 3-vector direction
    query_vector = [0.9] + [1.0] * (TEST_DIMENSION - 1)
    results = self.vector_store.semantic_search(query_vector, k=3)
    
    # Assert: The 3 nearest should all be from the 3-vector direction (ids 7-9)
    result_ids = [int(r['id']) for r in results]
    self.assertEqual(len(result_ids), 3)
    for result_id in result_ids:
      self.assertGreaterEqual(result_id, 7, f"Expected result from cluster (id >= 7), got {result_id}")

  def test_semantic_search_with_zero_k_raises_error(self):
    # Arrange
    query_vector = [1.0] * TEST_DIMENSION
    
    # Act & Assert: Should raise RuntimeError
    with self.assertRaises(RuntimeError) as context:
      self.vector_store.semantic_search(query_vector, k=0)
    
    self.assertIn("K must be at least 1", str(context.exception))

  def test_semantic_search_with_wrong_dimension_raises_error(self):
    # Arrange: Create a query vector with wrong dimension
    wrong_dimension_vector = [1.0] * (TEST_DIMENSION + 5)  # Wrong size
    
    # Act & Assert: Should raise RuntimeError
    with self.assertRaises(RuntimeError) as context:
      self.vector_store.semantic_search(wrong_dimension_vector, k=3)
    
    self.assertIn("dimension", str(context.exception).lower())

if __name__ == "__main__":
  unittest.main()
