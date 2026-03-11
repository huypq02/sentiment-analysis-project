"""
Module to initialize the main src.config package
Provides the imports at the package-level.
"""

from .dataclasses import (
    DataParameters, 
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking
)
from .constants import (
    VERSION,
    SERVICE_NAME,
    AUTHOR_NAME,
    AUTHOR_EMAIL,
    HEALTHY_STATUS,
    DEFAULT_CONFIG_PATH
)

__version__ = VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

__all__ = [
    "DataParameters", 
    "ComponentSelection",
    "Hyperparameters",
    "TrainingConfiguration",
    "FilePaths",
    "MLFlowTracking",
    "VERSION",
    "SERVICE_NAME",
    "AUTHOR_NAME",
    "AUTHOR_EMAIL",
    "HEALTHY_STATUS",
    "DEFAULT_CONFIG_PATH"
]
