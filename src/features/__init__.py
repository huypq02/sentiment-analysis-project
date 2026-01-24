"""
Module to initialize the package for src.features
Provides the imports at the package-level.
"""

__version__ = "1.2.0"
__author__ = "Huy Pham"
__email__ = "huypham0297@gmail.com"

from .base_feature_extractor import BaseFeatureExtractor
from .bow_extractor import BagOfWordsExtractor
from .tfidf_extractor import TFIDFExtractor

__all__ = ["BaseFeatureExtractor", "BagOfWordsExtractor", "TFIDFExtractor"]
