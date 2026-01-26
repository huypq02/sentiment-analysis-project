import unittest
from src import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def test_preprocess(self):
        """Unit test for the preprocessor."""
        preprocessor = Preprocessor()
        self.assertEqual(
            preprocessor.preprocess("You SHOULD TAKE this course!"), ["take", "course"]
        )        
        self.assertEqual(
            preprocessor.preprocess("The books are not bad!"), ["books", "not", "bad"]
        )


if __name__ == "__main__":
    unittest.main()
