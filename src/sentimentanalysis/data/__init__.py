"""
Module to initialize the package for sentimentanalysis.data
Provides the imports at the package-level.
"""

from sentimentanalysis.config import VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .data_loader import DataLoader
from .preprocessor import Preprocessor

__all__ = ["DataLoader", "Preprocessor"]
