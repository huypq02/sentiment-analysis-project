"""
Module to initialize the main src.config package
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

__all__ = [
    "DataParameters", 
    "ComponentSelection",
    "Hyperparameters",
    "TrainingConfiguration",
    "FilePaths",
    "MLFlowTracking"
]
