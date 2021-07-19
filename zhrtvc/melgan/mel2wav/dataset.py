import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from librosa.core import load
from librosa.util import normalize


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    curdir = Path(filename).parent
    outs = []
    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                fname = Path(line.strip().split("\t")[0])
                if fname.exists():
                    outs.append(fname)
                else:
                    fname = curdir.joinpath(fname)
                    if fname.exists():
                        outs.append(fname)
    return outs


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        # random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = augment

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0
        return audio.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(str(full_path), sr=None)
        if sampling_rate != self.sampling_rate:
            data = librosa.resample(data, sampling_rate, self.sampling_rate)

        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate
