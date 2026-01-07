"""
Module to initialize the package for src.pipeline
Provides the imports at the package-level.
"""

__version__ = '2.0.0'
__author__ = "Huy Pham"
__email__ = "huypham0297@gmail.com"

from .train_pipeline import train_main
from .predict_pipeline import predict_main
from .evaluation import evaluation_main
from .sentiment_classifier import SentimentClassifier

__all__ = ['train_main', 'predict_main', 'evaluation_main', 'SentimentClassifier']
