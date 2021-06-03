#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/12/10
"""
waveglow_train
"""
import json
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./waveglow/config.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument("--cuda", type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(Path(__file__).stem)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from waveglow.train import train

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    try:
        from setproctitle import setproctitle
        setproctitle('zhrtvc-waveglow-train')
    except ImportError:
        pass

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    # global data_config
    data_config = config["data_config"]
    # global dist_config
    dist_config = config["dist_config"]
    # global waveglow_config
    waveglow_config = config["waveglow_config"]

    metadata_dir = Path(train_config["output_directory"]).parent.joinpath('metadata')
    metadata_dir.mkdir(exist_ok=True, parents=True)

    copyfile(args.config, metadata_dir.joinpath('config.json'))
    # copyfile(data_config['training_files'], metadata_dir.joinpath('train.txt'))

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name,
          waveglow_config=waveglow_config,
          dist_config=dist_config,
          data_config=data_config,
          train_config=train_config,
          **train_config)


if __name__ == '__main__':
    main()
