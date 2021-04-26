#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/11/28
"""
melgan_inference
"""
from pathlib import Path
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Path(r'E:\githup\zhrtvc\data\best_step10_netG_torch.pt')
    parser.add_argument("-i", "--folder", type=Path, default=Path('../data/samples/aliaudio'),
                        help='输入音频文件的目录路径')
    parser.add_argument("-o", "--save_path", type=Path, default=Path("../data/results/melgan"),
                        help='输出生成语音的目录路径')
    parser.add_argument("-m", "--load_path", type=Path,
                        default=Path("../models/vocoder/saved_models/melgan/melgan_multi_speaker.pt"),
                        help='模型路径')
    parser.add_argument("--args_path", type=str, default='',
                        help='设置模型参数的文件')
    parser.add_argument("--mode", type=str, default='default',
                        help='模型模式')
    parser.add_argument("--n_samples", type=int, default=10,
                        help='需要实验多少个音频')
    parser.add_argument("--save_model_path", type=str, default='',
                        help='保存模型为可以直接torch.load的格式')
    parser.add_argument("--cuda", type=str, default='-1',
                        help='设置CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import traceback
import numpy as np

import librosa
import torch
import time
from tqdm import tqdm
from scipy.io import wavfile

from melgan.inference import MelVocoder, get_default_device, save_model

_device = get_default_device()


def main():
    vocoder = MelVocoder(args.load_path, github=args.mode == 'default', args_path=args.args_path,
                         device=_device, mode=args.mode)
    if args.save_model_path:
        save_model(vocoder, args.save_model_path)
    args.save_path.mkdir(exist_ok=True, parents=True)

    fpath_lst = [w for w in args.folder.glob("**/*") if w.is_file()]
    fpath_choices = np.random.choice(fpath_lst, min(args.n_samples, len(fpath_lst)), replace=False)
    for i, fname in enumerate(tqdm(fpath_choices, 'inference', ncols=100)):
        try:
            wav, sr = librosa.core.load(str(fname), sr=None)

            mel = vocoder(torch.from_numpy(wav[None]))

            recons = vocoder.inverse(mel.to(_device)).squeeze().cpu().numpy()

            strftime = time.strftime('%Y%m%d-%H%M%S')
            outdir = Path(args.save_path).joinpath(f'{args.load_path.stem}_{args.mode}')
            outdir.mkdir(exist_ok=True, parents=True)
            filename = str(outdir.joinpath(f'audio_{strftime}_{fname.stem}_raw.wav'))
            wavfile.write(filename=filename, rate=sr, data=wav)
            filename = str(outdir.joinpath(f'audio_{strftime}_{fname.stem}_syn.wav'))
            wavfile.write(filename=filename, rate=sr, data=recons)
            # librosa.output.write_wav(args.save_path.joinpath(f'{fname.stem}.wav'), recons, sr=sr)
        except:
            traceback.print_exc()


if __name__ == "__main__":
    main()
