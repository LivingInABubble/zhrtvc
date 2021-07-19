#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/12/10
"""
__init__
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    logger.info(__file__)
