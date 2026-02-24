from sklearn.feature_extraction.text import CountVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class BagOfWordsExtractor(BaseFeatureExtractor):
    def __init__(
            self,
            params: dict = {}
    ):
        """
        Initialize BagOfWordsExtractor with customizable parameters.

        Args:
            max_features (int or None): Maximum number of features to extract.
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams and bigrams).
            min_df (int or float): Ignore terms that appear in fewer than min_df documents (default: 1).
            max_df (int or float): Ignore terms that appear in more than max_df proportion of documents (default: 0.9).
            binary: Binary BoW captures presence, not frequency, which aligns better with sentiment signals (default: True).
            **vectorizer_kwargs: Additional keyword arguments for CountVectorizer.
        """
        self.vectorizer = CountVectorizer(**params)

    def fit(self, sentences):
        """
        Fit the Bag-of-Words extractor on training texts.

        Args:
            sentences: List of text documents to fit the vectorizer on.
        """
        self.vectorizer.fit(sentences)

    def transform(self, sentences):
        """
        Transform text into numerical representations.
        
        Args:
            sentences: List of text documents to transform.
        """
        return self.vectorizer.transform(sentences)
