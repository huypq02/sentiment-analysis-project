"""
Module to initialize the package for src.models
Provides the imports at the package-level.
"""

from src.config import API_VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = API_VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .model_interface import SentimentModel
from .logreg_model import LogisticRegressionModel
from .naive_bayes_model import NaiveBayesModel
from .factory import ModelFactory

__all__ = [
    "SentimentModel", 
    "LogisticRegressionModel", 
    "NaiveBayesModel", 
    "ModelFactory"
]
