from abc import ABC, abstractmethod

class BaseFeatureExtractor():
    """Abstract base class for feature extraction strategies."""

    @abstractmethod
    def fit(self, sentences):
        """Fit the extractor on training texts."""
        pass
    
    @abstractmethod
    def transform(self, sentences):
        """Transform text into numerical representations."""
        pass

    def fit_transform(self, sentences):
        """Fit the extractor and transform text into numerical representations."""
        pass