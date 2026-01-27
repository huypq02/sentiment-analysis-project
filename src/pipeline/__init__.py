"""
Module to initialize the package for src.pipeline
Provides the imports at the package-level.
"""

from src.config import API_VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = API_VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .training import train
from .prediction import predict
from .evaluation import evaluate
from .sentiment_pipeline import SentimentPipeline

__all__ = ["train", "predict", "evaluate", "SentimentPipeline"]
