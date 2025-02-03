import pyvista

# Start a virtual frame buffer
pyvista.start_xvfb()

import logging
import os
import warnings

#PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIRECTORY = os.getenv('PROJECT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
