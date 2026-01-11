from sklearn.feature_extraction.text import TfidfVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class TFIDFExtractor(BaseFeatureExtractor):
    def __init__(self):
        """Initialize the vectorizer for an extractor."""
        self.vectorizer = TfidfVectorizer()

    def fit(self, sentences):
        """Fit the extractor on training texts using TF-IDF."""
        self.vectorizer.fit(sentences)

    def transform(self, sentences):
        """Transform text into numerical representations after vectorization."""
        return self.vectorizer.transform(sentences)
