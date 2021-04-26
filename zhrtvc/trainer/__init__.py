# author: kuangdd
# date: 2021/4/26
"""
trainer
"""
from pathlib import Path
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    logger.info(__file__)
