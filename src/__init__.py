"""
Source code package for Text Classification project
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import preprocessing
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    'preprocessing',
    'models',
    'training',
    'evaluation',
    'utils'
]
