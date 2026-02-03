import logging
import os
import warnings
from pathlib import Path

from ._version import __version__

# Get the project root directory (two levels up from this __init__.py file)
# This file is at: src/pyVertexModel/__init__.py
# We want the project root directory (the parent of 'src')
PROJECT_DIRECTORY = os.getenv('PROJECT_DIR', str(Path(__file__).parent.parent.parent))

# get the logger instance
logger = logging.getLogger("pyVertexModel")

formatter = logging.Formatter(
    "%(levelname)s [%(asctime)s] pyVertexModel: %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
logger.propagate = False


# Function to handle warnings
def warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f'{filename}:{lineno}: {category.__name__}: {message}')


# Set the warnings' showwarning function to the handler
warnings.showwarning = warning_handler

__all__ = ['__version__', 'PROJECT_DIRECTORY', 'logger']