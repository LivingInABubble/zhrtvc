#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/20
"""
"""

import json

import numpy as np
import torch
import yaml

from .mel2wav.interface import MelVocoder, get_default_device
from .mel2wav.modules import Generator

_model = None


def load_melgan_model(model_path, args_path, device=None):
    """
    导入训练得到的checkpoint模型文件。
    """
    global _model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if str(args_path).endswith('.yml'):
        with open(args_path, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
    else:
        args = json.load(open(args_path, encoding='utf8'))

    ratios = [int(w) for w in args['ratios'].split()]
    _model = Generator(args['n_mel_channels'], args['ngf'], args['n_residual_layers'], ratios=ratios).to(device)
    _model.load_state_dict(torch.load(model_path, map_location=device))
    return _model


def load_melgan_torch(model_path, device=None):
    """
    用torch.load直接导入模型文件，不需要导入模型代码。
    """
    global _model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = torch.load(model_path, map_location=device)
    return _model


def is_loaded():
    """
    判断模型是否已经被导入。
    """
    global _model
    return _model is not None


def generate_wave(mel, **kwargs):
    """
    用声码器模型把mel频谱转为语音信号。
    """
    global _model
    if not is_loaded():
        load_melgan_torch(**kwargs)
    with torch.no_grad():
        wav = _model(mel)
        return wav


_melgan_vocoder = None


def load_vocoder_melgan(load_path):
    global _melgan_vocoder
    if _melgan_vocoder is None:
        _melgan_vocoder = MelVocoder(load_path, github=True)
    return _melgan_vocoder


def infer_waveform_melgan(mel, load_path=None):
    global _melgan_vocoder
    if _melgan_vocoder is None:
        _melgan_vocoder = MelVocoder(load_path, github=True)

    mel = torch.from_numpy(mel[np.newaxis].astype(np.float32))
    wav = _melgan_vocoder.inverse(mel).squeeze().cpu().numpy()
    return wav


_net_generator = None


def mel2wav_melgan(mel, load_path=None, device=get_default_device()):
    global _net_generator
    if _net_generator is None:
        _net_generator = torch.load(load_path, map_location=device)
    with torch.no_grad():
        return _net_generator(mel.to(device)).squeeze(1)


def save_model(model: MelVocoder, outpath):
    torch.save(model.mel2wav_model, outpath)


if __name__ == "__main__":
    print(__file__)
