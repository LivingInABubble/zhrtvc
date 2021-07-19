import numpy as np
from aukit.audio_griffinlim import default_hparams
from aukit.audio_griffinlim import inv_mel_spectrogram
from aukit.audio_griffinlim import load_wav
from aukit.audio_io import Dict2Obj
from aukit.audio_spectrogram import mel_spectrogram

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
    '__file__': __file__
}

melgan_hparams = {}
melgan_hparams.update({k: v for k, v in default_hparams.items()})
melgan_hparams.update(my_hp)
melgan_hparams = Dict2Obj(melgan_hparams)

_pad_len = (default_hparams.n_fft - default_hparams.hop_size) // 2


def melspectrogram(wav, hparams=None):
    wav = np.pad(wav.flatten(), (_pad_len, _pad_len), mode="reflect")
    mel = mel_spectrogram(wav, melgan_hparams)
    mel = mel / 20
    return mel


def inv_melspectrogram(mel, hparams=None):
    mel = mel * 20
    wav = inv_mel_spectrogram(mel, melgan_hparams)
    return wav


if __name__ == "__main__":
    import aukit

    inpath = r"E:\data\temp\01.wav"
    wav = load_wav(inpath, sr=16000)
    mel = melspectrogram(wav)
    out = inv_melspectrogram(mel)
    aukit.play_audio(wav)
    aukit.play_audio(out)
