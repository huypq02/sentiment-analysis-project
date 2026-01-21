"""
Module to initialize the package for src.utils
Provides the imports at the package-level.
"""

__version__ = "2.0.0"
__author__ = "Huy Pham"
__email__ = "huypham0297@gmail.com"

from .config import config
from .logger import setup_logging
from .sentiment_mapper import rating_to_sentiment

__all__ = [
    "config", 
    "setup_logging", 
    "rating_to_sentiment"
]
