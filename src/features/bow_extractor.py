from sklearn.feature_extraction.text import CountVectorizer
from .base_feature_extractor import BaseFeatureExtractor

class BagOfWordsExtractor(BaseFeatureExtractor):
    def __init__(self, max_features=None, min_df=1, max_df=1.0, stop_words=None, **vectorizer_kwargs):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            **vectorizer_kwargs
        )
    
    def fit(self, sentences):
        """Fit the Bag-of-Words extractor on training texts."""
        self.vectorizer.fit(sentences)
    
    def transform(self, sentences):
        """Transform text into numerical representations."""
        return self.vectorizer.transform(sentences)