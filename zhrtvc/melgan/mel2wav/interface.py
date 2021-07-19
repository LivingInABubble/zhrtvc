import json
from pathlib import Path

import numpy as np
import torch
import yaml
from aukit import Dict2Obj
from aukit.audio_griffinlim import mel_spectrogram, default_hparams

from .modules import Generator, Audio2Mel


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(mel2wav_path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    root = Path(mel2wav_path)
    with open(root / "args.yml", "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    netG.load_state_dict(torch.load(root / "best_netG.pt", map_location=device))
    return netG


def load_state_dict(pt_path, yml_path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    if str(yml_path).endswith('.yml'):
        with open(yml_path, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
    else:
        args = json.load(open(yml_path, encoding='utf8'))

    ratios = [int(w) for w in args['ratios'].split()]
    netG = Generator(args['n_mel_channels'], args['ngf'], args['n_residual_layers'], ratios=ratios).to(device)
    netG.load_state_dict(torch.load(pt_path, map_location=device))
    return netG


def load_model_net(load_path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    netG = torch.load(load_path, map_location=device)
    return netG


class MelVocoder:
    def __init__(
            self,
            path,
            device=get_default_device(),
            github=False,
            args_path="",
            mode='default',
    ):
        if mode == 'default':
            self.fft = audio2mel
        elif mode == 'synthesizer':
            self.fft = audio2mel_synthesizer
        elif mode == 'mellotron':
            self.fft = audio2mel_mellotron
        else:
            raise KeyError

        if github:
            netG = Generator(80, 32, 3).to(device)
            netG.load_state_dict(
                torch.load(path, map_location=device)
            )
            self.mel2wav_model = netG
        else:
            if args_path:
                self.mel2wav_model = load_state_dict(path, args_path, device)
            else:
                self.mel2wav_model = load_model_net(path, device)

        self.device = device

    def __call__(self, audio):
        """
        Performs audio to mel conversion (See Audio2Mel in mel2wav/modules.py)
        Args:
            audio (torch.tensor): PyTorch tensor containing audio (batch_size, timesteps)
        Returns:
            torch.tensor: log-mel-spectrogram computed on input audio (batch_size, 80, timesteps)
        """
        return self.fft(audio)

    def inverse(self, mel):
        """
        Performs mel2audio conversion
        Args:
            mel (torch.tensor): PyTorch tensor containing log-mel spectrograms (batch_size, 80, timesteps)
        Returns:
            torch.tensor:  Inverted raw audio (batch_size, timesteps)

        """
        with torch.no_grad():
            return self.mel2wav_model(mel.to(self.device)).squeeze(1)

    def wav2mel(self, wav):
        return self.__call__(wav)

    def mel2wav(self, mel):
        return self.inverse(mel)


def audio2mel(src):
    """
    训练中使用。
    :param src:
    :return:
    """
    src = src.unsqueeze(1)
    mel = Audio2Mel()(src)
    return mel


_sr = 22050
my_hp = {
    "n_fft": 1024,  # 800
    "hop_size": 256,  # 200
    "win_size": 1024,  # 800
    "sample_rate": _sr,  # 16000
    "fmin": 0,  # 55
    "fmax": _sr // 2,  # 7600
    "preemphasize": False,  # True
    'symmetric_mels': True,  # True
    'signal_normalization': False,  # True
    'allow_clipping_in_normalization': False,  # True
    'ref_level_db': 0,  # 20
    'center': False,  # True
    '__file__': __file__
}

synthesizer_hparams = {k: v for k, v in default_hparams.items()}
synthesizer_hparams = {**synthesizer_hparams, **my_hp}
synthesizer_hparams = Dict2Obj(synthesizer_hparams)


def audio2mel_synthesizer(src):
    """
    用aukit模块重现生成mel，和synthesizer的频谱适应。
    :param src:
    :return:
    """
    _pad_len = (synthesizer_hparams.n_fft - synthesizer_hparams.hop_size) // 2
    wavs = src.cpu().numpy()
    mels = []
    for wav in wavs:
        wav = np.pad(wav.flatten(), (_pad_len, _pad_len), mode="reflect")
        mel = mel_spectrogram(wav, synthesizer_hparams)
        mel = mel / 20
        mels.append(mel)
    mels = torch.from_numpy(np.array(mels).astype(np.float32))
    return mels


def audio2mel_mellotron(src):
    """
    用aukit模块重现生成mel，和mellotron的频谱适应。
    :param src:
    :return:
    """
    wavs = src.cpu().numpy()
    mels = []
    for wav in wavs:
        wav = wav.flatten()[:-1]  # 避免生成多一个空帧频谱
        mel = mel_spectrogram(wav, default_hparams)
        mels.append(mel)
    mels = torch.from_numpy(np.array(mels).astype(np.float32))
    return mels
