# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/8/13
"""
__init__
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    print(__file__)
