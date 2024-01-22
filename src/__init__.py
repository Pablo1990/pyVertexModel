import logging
import os

PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

