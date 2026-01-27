"""
Module to initialize the package for src.data
Provides the imports at the package-level.
"""

from src.config import API_VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = API_VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .data_loader import DataLoader
from .preprocessor import Preprocessor

__all__ = ["DataLoader", "Preprocessor"]
