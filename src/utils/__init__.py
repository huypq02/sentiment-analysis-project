"""
Module to initialize the package for src.utils
Provides the imports at the package-level.
"""

from src.config import API_VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = API_VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .config import load_config
from .logger import setup_logging
from .sentiment_mapper import rating_to_sentiment

__all__ = [
    "load_config", 
    "setup_logging", 
    "rating_to_sentiment"
]
