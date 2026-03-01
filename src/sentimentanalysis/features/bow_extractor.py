from sklearn.feature_extraction.text import CountVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class BagOfWordsExtractor(BaseFeatureExtractor):
    def __init__(
            self,
            params: dict = {}
    ):
        """
        Initialize BagOfWordsExtractor with customizable parameters.

        :param params: Parameters for CountVectorizer (max_features, ngram_range, min_df, max_df, binary, etc.)
        :type params: dict
        """
        self.vectorizer = CountVectorizer(**params)

    def fit(self, sentences):
        """
        Fit the Bag-of-Words extractor on training texts.

        :param sentences: List of text documents to fit the vectorizer on
        :type sentences: list[str]
        """
        self.vectorizer.fit(sentences)

    def transform(self, sentences):
        """
        Transform text into numerical representations.
        
        :param sentences: List of text documents to transform
        :type sentences: list[str]
        :return: Transformed bag-of-words features
        :rtype: scipy.sparse.csr_matrix
        """
        return self.vectorizer.transform(sentences)
