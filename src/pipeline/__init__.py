"""
Module to initialize the package for src.pipeline
Provides the imports at the package-level.
"""

__version__ = "2.0.0"
__author__ = "Huy Pham"
__email__ = "huypham0297@gmail.com"

from .training import train
from .prediction import predict
from .evaluation import evaluate
from .sentiment_pipeline import SentimentPipeline

__all__ = ["train", "predict", "evaluate", "SentimentPipeline"]
