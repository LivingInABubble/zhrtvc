#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/23
"""
"""
import os
import sys

# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # from utils.argutils import print_args
    from melgan.train import train_melgan, parse_args

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    try:
        from setproctitle import setproctitle

        setproctitle('zhrtvc-melgan-train')
    except ImportError:
        pass

    # print_args(args, parser)
    train_melgan(args)


if __name__ == '__main__':
    main()
