from sklearn.feature_extraction.text import TfidfVectorizer
from .base_feature_extractor import BaseFeatureExtractor


class TFIDFExtractor(BaseFeatureExtractor):
    def __init__(
            self, 
            params: dict = {}
    ):
        """
        Initialize the vectorizer for an extractor.
        
        :param params: Parameters for TfidfVectorizer (max_features, ngram_range, min_df, max_df, sublinear_tf, etc.)
        :type params: dict
        """
        self.vectorizer = TfidfVectorizer(**params)

    def fit(self, sentences):
        """
        Fit the extractor on training texts using TF-IDF.

        :param sentences: List of text documents to fit the vectorizer on
        :type sentences: list[str]
        """
        self.vectorizer.fit(sentences)

    def transform(self, sentences):
        """
        Transform text into numerical representations after vectorization.
        
        :param sentences: List of text documents to transform
        :type sentences: list[str]
        :return: Transformed TF-IDF features
        :rtype: scipy.sparse.csr_matrix
        """
        return self.vectorizer.transform(sentences)
