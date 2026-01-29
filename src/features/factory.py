from typing import Optional
from . import TFIDFExtractor, BagOfWordsExtractor

class ExtractorFactory:
    @staticmethod
    def create_extractor(extractor_name: str, params: Optional[dict]):
        """
        Create a feature extractor.

        Args:
            extractor_name (str): The name of the extractor to create. Supported values are 'tfidf' and 'bow'.
            params (Optional[dict]): Parameters to initialize the extractor. If None, default parameters are used.

        Returns:
            An instance of the specified feature extractor.

        Raises:
            ValueError: If an unknown extractor_name is provided.
        """
        if extractor_name == "tfidf":
            if params is None:
                params = {
                    "max_features": 5000,
                    "ngram_range": (1, 2),  # Unigrams + bigrams to capture phrases like "not bad"
                    "min_df": 2,
                    "max_df": 0.9
                }
            return TFIDFExtractor(**params)
        elif extractor_name == "bow":
            if params is None:
                params = {
                    "max_features": 5000,
                    "ngram_range": (1, 2),  # Unigrams + bigrams to capture phrases like "not bad"
                    "min_df": 2,
                    "max_df": 0.8
                }
            return BagOfWordsExtractor(**params)
        else:
            raise ValueError(f"Unknown feature extractor name: {extractor_name}")
