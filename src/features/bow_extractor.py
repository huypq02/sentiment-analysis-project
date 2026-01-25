from sklearn.feature_extraction.text import CountVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class BagOfWordsExtractor(BaseFeatureExtractor):
    def __init__(
            self,
            max_features=5000,
            ngram_range=(1,2),
            min_df=2,
            max_df=0.8,
            binary=True,
            **vectorizer_kwargs
    ):
        """
        Initialize BagOfWordsExtractor with customizable parameters.

        Args:
            max_features (int or None): Maximum number of features to extract.
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams and bigrams).
            min_df (int or float): Ignore terms that appear in fewer than min_df documents.
            max_df (int or float): Ignore terms that appear in more than max_df proportion of documents.
            binary: Binary BoW captures presence, not frequency, which aligns better with sentiment signals.
            **vectorizer_kwargs: Additional keyword arguments for CountVectorizer.
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            **vectorizer_kwargs
        )

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
