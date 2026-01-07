"""
Module to initialize the package for src.models
Provides the imports at the package-level.
"""

__version__ = '2.0.0'
__author__ = "Huy Pham"
__email__ = "huypham0297@gmail.com"

from .model_interface import SentimentModel
from .logreg_model import LogisticRegressionModel
from .naive_bayes_model import NaiveBayesModel

__all__ = ['SentimentModel', 'LogisticRegressionModel', 'NaiveBayesModel']
