from . import TFIDFExtractor, BagOfWordsExtractor

class ExtractorFactory:
    @staticmethod
    def create_extractor(extractor_name: str):
        """Create a feature extractor."""
        if extractor_name == "tfidf":
            return TFIDFExtractor(
                max_features=5000,
                ngram_range=(1, 2),  # Unigrams + bigrams to capture phrases like "not bad"
                min_df=2,
                max_df=0.9
            )
        elif extractor_name == "bow":
            return BagOfWordsExtractor(
                max_features=5000,
                ngram_range=(1,2),
                min_df=2,
                max_df=0.8,
            )
        else:
            raise ValueError(f"Unknown feature extractor name: {extractor_name}")
