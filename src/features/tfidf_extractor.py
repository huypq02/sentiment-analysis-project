from sklearn.feature_extraction.text import TfidfVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class TFIDFExtractor(BaseFeatureExtractor):
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9):
        """Initialize the vectorizer for an extractor.
        
        Args:
            max_features: Maximum number of features (default: 5000)
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams and bigrams)
            min_df: Ignore terms appearing in fewer documents (default: 2)
            max_df: Ignore terms appearing in more than this proportion of docs (default: 0.9)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,  # Capture phrases like "not bad", "very good"
            min_df=min_df,            # Remove very rare words
            max_df=max_df,            # Remove very common words
            sublinear_tf=True         # Use logarithmic scaling for term frequency
        )

    def fit(self, sentences):
        """Fit the extractor on training texts using TF-IDF."""
        self.vectorizer.fit(sentences)

    def transform(self, sentences):
        """Transform text into numerical representations after vectorization."""
        return self.vectorizer.transform(sentences)
