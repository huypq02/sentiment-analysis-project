"""
Module to initialize the main src package
Provides the imports at the package-level.
"""

from .config import (
    DataParameters, 
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking,
    API_VERSION,
    SERVICE_NAME,
    AUTHOR_NAME,
    AUTHOR_EMAIL
)
from .data import DataLoader, Preprocessor
from .features import BagOfWordsExtractor, TFIDFExtractor, ExtractorFactory
from .models import LogisticRegressionModel, NaiveBayesModel, ModelFactory
from .pipeline import train, predict, evaluate, SentimentPipeline
from .utils import load_config, setup_logging

__version__ = API_VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

__all__ = [
    "DataLoader",
    "Preprocessor",
    "BagOfWordsExtractor",
    "TFIDFExtractor",
    "LogisticRegressionModel",
    "NaiveBayesModel",
    "train",
    "predict",
    "evaluate",
    "SentimentPipeline",
    "load_config",
    "setup_logging",
    "ExtractorFactory",
    "ModelFactory",
    "DataParameters", 
    "ComponentSelection",
    "Hyperparameters",
    "TrainingConfiguration",
    "FilePaths",
    "MLFlowTracking",
    "API_VERSION",
    "SERVICE_NAME",
    "AUTHOR_NAME",
    "AUTHOR_EMAIL"
]
