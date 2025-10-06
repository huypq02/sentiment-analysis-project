from sklearn.feature_extraction.text import CountVectorizer
from .base_feature_extractor import BaseFeatureExtractor

class BagOfWordsExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.vectorizer = CountVectorizer()
    
    def fit(self, sentences):
        """Fit the Bag-of-Words extractor on training texts."""
        self.vectorizer.fit(sentences)
    
    def transform(self, sentences):
        """Transform text into numerical representations."""
        return self.vectorizer.transform(sentences)