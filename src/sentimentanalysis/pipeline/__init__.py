"""
Module to initialize the package for sentimentanalysis.pipeline
Provides the imports at the package-level.
"""

from sentimentanalysis.config import VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .training import train
from .prediction import predict
from .evaluation import evaluate, evaluate_saved_model
from .sentiment_pipeline import SentimentPipeline

__all__ = ["train", "predict", "evaluate", "evaluate_saved_model", "SentimentPipeline"]
