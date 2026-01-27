"""
Module to initialize the package for src.features
Provides the imports at the package-level.
"""

from src.config import API_VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = API_VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .base_feature_extractor import BaseFeatureExtractor
from .bow_extractor import BagOfWordsExtractor
from .tfidf_extractor import TFIDFExtractor
from .factory import ExtractorFactory

__all__ = [
    "BaseFeatureExtractor", 
    "BagOfWordsExtractor", 
    "TFIDFExtractor", 
    "ExtractorFactory"
]
