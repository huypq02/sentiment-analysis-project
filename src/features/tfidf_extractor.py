from sklearn.feature_extraction.text import TfidfVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class TFIDFExtractor(BaseFeatureExtractor):
    def __init__(
            self, 
            params: dict = {}
    ):
        """Initialize the vectorizer for an extractor.
        
        Args:
            max_features: Maximum number of features (default: 5000).
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams and bigrams).
            min_df: Ignore terms appearing in fewer documents (default: 1).
            max_df: Ignore terms appearing in more than this proportion of docs (default: 0.9).
            sublinear_tf: Sublinear tf scaling for term frequency (default: True).
            **vectorizer_kwargs: Additional keyword arguments for TfidfVectorizer.
        """
        self.vectorizer = TfidfVectorizer(**params)

    def fit(self, sentences):
        """
        Fit the extractor on training texts using TF-IDF.

        Args:
            sentences: List of text documents to fit the vectorizer on.
        """
        self.vectorizer.fit(sentences)

    def transform(self, sentences):
        """
        Transform text into numerical representations after vectorization.
        
        Args:
            sentences: List of text documents to transform.
        """
        return self.vectorizer.transform(sentences)
