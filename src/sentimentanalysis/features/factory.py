from typing import Optional
from . import TFIDFExtractor, BagOfWordsExtractor

class ExtractorFactory:
    @staticmethod
    def create_extractor(extractor_name: str, params: Optional[dict]):
        """
        Create a feature extractor.

        :param extractor_name: The name of the extractor to create (Supported: 'tfidf', 'bow')
        :type extractor_name: str
        :param params: Parameters to initialize the extractor (If None, default parameters are used)
        :type params: Optional[dict]
        :return: An instance of the specified feature extractor
        :rtype: BaseFeatureExtractor
        :raises ValueError: If an unknown extractor_name is provided
        """
        if extractor_name == "tfidf":
            if not params:
                return TFIDFExtractor()
            return TFIDFExtractor(params)
        elif extractor_name == "bow":
            if not params:
                return BagOfWordsExtractor()
            return BagOfWordsExtractor(params)
        else:
            raise ValueError(f"Unknown feature extractor name: {extractor_name}")
