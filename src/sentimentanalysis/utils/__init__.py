"""
Module to initialize the package for sentimentanalysis.utils
Provides the imports at the package-level.
"""

from sentimentanalysis.config import VERSION, AUTHOR_NAME, AUTHOR_EMAIL

__version__ = VERSION
__author__ = AUTHOR_NAME
__email__ = AUTHOR_EMAIL

from .config import load_config
from .logger import setup_logging


__all__ = [
    "load_config", 
    "setup_logging"
]
