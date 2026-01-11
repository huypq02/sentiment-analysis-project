from sklearn.feature_extraction.text import CountVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class BagOfWordsExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        max_features=None,
        min_df=1,
        max_df=1.0,
        stop_words=None,
        **vectorizer_kwargs
    ):
        """
        Initialize BagOfWordsExtractor with customizable parameters.

        Args:
            max_features (int or None): Maximum number of features to extract. None means unlimited.
            min_df (int or float): Ignore terms that appear in fewer than min_df documents.
            max_df (int or float): Ignore terms that appear in more than max_df proportion of documents.
            stop_words (str, list, or None): Stop words to remove. None means do not remove stop words.
            **vectorizer_kwargs: Additional keyword arguments for CountVectorizer.

        By default:
            - Uses all features (max_features=None).
            - Removes words that appear in fewer than 1 document (min_df=1) or more than 100% of documents (max_df=1.0).
            - Does not ignore stopwords (stop_words=None).
        """
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
