"""
Module to initialize the package for src.utils
Provides the imports at the package-level.
"""

__version__ = "2.0.0"
__author__ = "Huy Pham"
__email__ = "huypham0297@gmail.com"

from .load_config import load_config
from .logger import setup_logging

__all__ = ["load_config", "setup_logging"]
