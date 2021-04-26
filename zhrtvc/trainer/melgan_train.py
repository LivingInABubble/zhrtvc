#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/23
"""
"""
from pathlib import Path
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.argutils import print_args
from melgan.train import train_melgan, parse_args

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

if __name__ == "__main__":
    try:
        from setproctitle import setproctitle

        setproctitle('zhrtvc-melgan-train')
    except ImportError:
        pass

    # print_args(args, parser)
    train_melgan(args)
