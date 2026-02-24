import unittest
from sentimentanalysis import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def test_preprocess(self):
        """Unit test for the preprocessor."""
        preprocessor = Preprocessor()
        self.assertEqual(
            preprocessor.preprocess("You SHOULD TAKE this course!"), ["take", "course"]
        )        
        self.assertEqual(
            preprocessor.preprocess("The movie is not bad!"), ["movie", "not", "bad"]
        )


if __name__ == "__main__":
    unittest.main()
