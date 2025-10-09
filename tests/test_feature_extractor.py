import unittest
import numpy as np
from src.features.bow_extractor import BagOfWordsExtractor
from src.features.tfidf_extractor import TFIDFExtractor

class TestBagOfWordsExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = BagOfWordsExtractor()
        # Dummy data
        self.sentences = [
            "The machine learning algorithm performed exceptionally well on the validation dataset.",
            "Despite the challenging weather conditions, the autonomous vehicle successfully navigated through the busy intersection.",
            "Natural language processing techniques have revolutionized how we analyze and understand human communication patterns.",
            "The convolutional neural network architecture demonstrated superior performance compared to traditional computer vision methods.",
            "Deep learning models require substantial computational resources and carefully curated training datasets to achieve optimal results."
        ]
    
    def test_fit(self):
        """Unit test for fitting the extractor on training texts."""
        self.extractor.fit(self.sentences)
        self.assertGreater(len(self.extractor.vectorizer.get_feature_names_out()), 0)

    def test_transform(self):
        """Unit test for transforming text into numerical representations."""
        X = self.extractor.fit_transform(self.sentences)
        # Check shape
        self.assertEqual(X.shape[0], len(self.sentences))                                    # rows
        self.assertEqual(X.shape[1], len(self.extractor.vectorizer.get_feature_names_out())) # columns
        # Check data type
        self.assertTrue(hasattr(X, "toarray"))
                        
class TestTFIDFExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = TFIDFExtractor()
        # Dummy data
        self.sentences = [
            "The machine learning algorithm performed exceptionally well on the validation dataset.",
            "Despite the challenging weather conditions, the autonomous vehicle successfully navigated through the busy intersection.",
            "Natural language processing techniques have revolutionized how we analyze and understand human communication patterns.",
            "The convolutional neural network architecture demonstrated superior performance compared to traditional computer vision methods.",
            "Deep learning models require substantial computational resources and carefully curated training datasets to achieve optimal results."
        ]
    def test_fit(self):
        """Unit test for fitting the extractor on training texts."""
        self.extractor.fit(self.sentences)
        self.assertGreater(len(self.extractor.vectorizer.get_feature_names_out()), 0)

    def test_transform(self):
        """Unit test for transforming text into numerical representations."""
        X = self.extractor.fit_transform(self.sentences)
        # Check shape
        self.assertTrue(X[0].shape, len(self.sentences))                                    # rows
        self.assertTrue(X[1].shape, len(self.extractor.vectorizer.get_feature_names_out())) # columns
                        
if __name__ == "__main__":
    unittest.main()