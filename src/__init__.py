"""
Module to initialize the main src package
Provides the imports at the package-level.
"""

__version__ = "1.2.0"
__author__ = "Huy Pham"
__email__ = "huypham0297@gmail.com"

from .config import (
    DataParameters, 
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking
)
from .data import DataLoader, Preprocessor
from .features import BagOfWordsExtractor, TFIDFExtractor, ExtractorFactory
from .models import LogisticRegressionModel, NaiveBayesModel, ModelFactory
from .pipeline import train, predict, evaluate, SentimentPipeline
from .utils import load_config, setup_logging

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
    "MLFlowTracking"
]
