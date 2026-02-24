from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction strategies."""

    @abstractmethod
    def fit(self, sentences):
        """
        Fit the extractor on training texts.
        
        :param sentences: List of text documents
        :type sentences: list[str]
        """
        pass

    @abstractmethod
    def transform(self, sentences):
        """
        Transform text into numerical representations.
        
        :param sentences: List of text documents
        :type sentences: list[str]
        :return: Transformed features
        :rtype: array-like
        """
        pass

    def fit_transform(self, sentences):
        """
        Fit the extractor and transform text into numerical representations.
        
        :param sentences: List of text documents
        :type sentences: list[str]
        :return: Transformed features
        :rtype: array-like
        """
        self.fit(sentences)
        return self.transform(sentences)
