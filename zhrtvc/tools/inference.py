#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/4/6
"""
"""
import json
import time


def get_ipynb():
    inpath = r"E:\lab\mellotron\mellotron-master\inference.ipynb"
    srcdt = json.load(open(inpath, encoding="utf8"))
    cells = srcdt["cells"]
    for celldt in cells:
        lines = celldt["source"]
        out = "".join(lines)
        print(out)


import matplotlib.pyplot as plt
import IPython.display as ipd

import sys

sys.path.append('waveglow/')

from itertools import cycle
import numpy as np
from scipy.io.wavfile import write
import pandas as pd
import librosa
import torch

from hparams import create_hparams
from model import load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
from mellotron_utils import get_data_from_musicxml
from melgan.inference import load_vocoder_melgan, infer_waveform_melgan

import aukit


def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2) / 2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2) / 2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]


def plot_mel_f0_alignment(mel_source, mel_outputs_postnet, f0s, alignments, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto', origin='bottom', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()


def load_mel(path):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec


hparams = create_hparams()
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)

model_path = "models/mellotron_libritts.pt"
mellotron = load_model(hparams).cuda().eval()
mellotron.load_state_dict(torch.load(model_path)['state_dict'])

waveglow_path = 'models/waveglow_256channels_v4.pt'
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()

melgan_path = 'models/multi_speaker.pt'
load_vocoder_melgan(melgan_path)

## Setup dataloaders
arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
audio_paths = 'data/examples_filelist.txt'
dataloader = TextMelLoader(audio_paths, hparams)
datacollate = TextMelCollate(1)

## Load data
file_idx = 0
audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]

# get audio path, encoded text, pitch contour and mel for gst
text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
pitch_contour = dataloader[file_idx][3][None].cuda()
mel = load_mel(audio_path)
print(audio_path, text)

## Define Speakers Set
speaker_ids = TextMelLoader("filelists/libritts_train_clean_100_audiopath_text_sid_atleast5min_val_filelist.txt",
                            hparams).speaker_ids
speakers = pd.read_csv('filelists/libritts_speakerinfo.txt', engine='python', header=None, comment=';', sep=' *\| *',
                       names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'])
speakers['MELLOTRON_ID'] = speakers['ID'].apply(lambda x: speaker_ids[x] if x in speaker_ids else -1)
female_speakers = cycle(
    speakers.query("SEX == 'F' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())
male_speakers = cycle(
    speakers.query("SEX == 'M' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())


def style_transfer():
    # load source data to obtain rhythm using tacotron 2 as a forced aligner
    x, y = mellotron.parse_batch(datacollate([dataloader[file_idx]]))
    ipd.Audio(audio_path, rate=hparams.sampling_rate)

    # Style Transfer (Rhythm and Pitch Contour)
    with torch.no_grad():
        # get rhythm (alignment map) using tacotron 2
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.forward(x)
        rhythm = rhythm.permute(1, 0, 2)
    speaker_id = next(female_speakers) if np.random.randint(2) else next(male_speakers)
    speaker_id = torch.LongTensor([speaker_id]).cuda()

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
            (text_encoded, mel, speaker_id, pitch_contour, rhythm))

    plot_mel_f0_alignment(x[2].data.cpu().numpy()[0],
                          mel_outputs_postnet.data.cpu().numpy()[0],
                          pitch_contour.data.cpu().numpy()[0, 0],
                          rhythm.data.cpu().numpy()[:, 0].T)
    plt.show()

    out_mel = mel_outputs_postnet.data.cpu().numpy()[0]

    # wav = aukit.inv_mel_spectrogram()
    out_wav = infer_waveform_melgan(out_mel)

    aukit.play_audio(out_wav, sr=22050)

    with torch.no_grad():
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    out_wav = audio[0].data.cpu().numpy()

    aukit.play_audio(out_wav, sr=22050)


def singing_voice():
    # Singing Voice from Music Score
    data = get_data_from_musicxml('data/haendel_hallelujah.musicxml', 132, convert_stress=True)
    panning = {'Soprano': [-60, -30], 'Alto': [-40, -10], 'Tenor': [30, 60], 'Bass': [10, 40]}
    n_speakers_per_part = 4
    frequency_scaling = 0.4
    n_seconds = 90
    audio_stereo = np.zeros((hparams.sampling_rate * n_seconds, 2), dtype=np.float32)
    for i, (part, v) in enumerate(data.items()):
        rhythm = data[part]['rhythm'].cuda()
        pitch_contour = data[part]['pitch_contour'].cuda()
        text_encoded = data[part]['text_encoded'].cuda()

        for k in range(n_speakers_per_part):
            pan = np.random.randint(panning[part][0], panning[part][1])
            if any(x in part.lower() for x in ('soprano', 'alto', 'female')):
                speaker_id = torch.LongTensor([next(female_speakers)]).cuda()
            else:
                speaker_id = torch.LongTensor([next(male_speakers)]).cuda()
            print("{} MellotronID {} pan {}".format(part, speaker_id.item(), pan))

            with torch.no_grad():
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments_transfer = mellotron.inference_noattention(
                    (text_encoded, mel, speaker_id, pitch_contour * frequency_scaling, rhythm))

            plot_mel_f0_alignment(mel_outputs_postnet.data.cpu().numpy()[0],
                                  mel_outputs_postnet.data.cpu().numpy()[0],
                                  pitch_contour.data.cpu().numpy()[0, 0],
                                  rhythm.data.cpu().numpy()[:, 0].T)
            plt.show()
            out_mel = mel_outputs_postnet.data.cpu().numpy()[0]

            # wav = aukit.inv_mel_spectrogram()
            out_wav = infer_waveform_melgan(out_mel)

            aukit.play_audio(out_wav, sr=22050)

            audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
            audio = audio.cpu().numpy()
            audio = panner(audio, pan)
            audio_stereo[:audio.shape[0]] += audio
            write("{} {}.wav".format(part, speaker_id.item()), hparams.sampling_rate, audio)
            out_wav = audio

            aukit.play_audio(out_wav, sr=22050)

    audio_stereo = audio_stereo / np.max(np.abs(audio_stereo))
    write("audio_stereo.wav", hparams.sampling_rate, audio_stereo)
    ipd.Audio([audio_stereo[:, 0], audio_stereo[:, 1]], rate=hparams.sampling_rate)


def style_transfer_v2():
    audio_paths_ = 'data/examples_filelist_v2.txt'
    dataloader_ = TextMelLoader(audio_paths_, hparams)
    datacollate_ = TextMelCollate(1)
    ## Load data
    # for file_idx in range(10):
    #     audio_path, text, sid = dataloader_.audiopaths_and_text[file_idx]
    #     print(dict(file_idx=file_idx, audio_path=audio_path, text=text))

    file_idx = 8
    audio_path, text, sid = dataloader_.audiopaths_and_text[file_idx]
    print(dict(file_idx=file_idx, audio_path=audio_path, text=text, sid=sid))

    # get audio path, encoded text, pitch contour and mel for gst
    text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
    pitch_contour = dataloader_[file_idx][3][None].cuda()
    mel = load_mel(audio_path)

    # load source data to obtain rhythm using tacotron 2 as a forced aligner
    x, y = mellotron.parse_batch(datacollate_([dataloader_[file_idx]]))
    ipd.Audio(audio_path, rate=hparams.sampling_rate)

    # Style Transfer (Rhythm and Pitch Contour)
    with torch.no_grad():
        # get rhythm (alignment map) using tacotron 2
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.forward(x)
        rhythm = rhythm.permute(1, 0, 2)
    speaker_id = next(female_speakers) if np.random.randint(2) else next(male_speakers)
    speaker_id = torch.LongTensor([speaker_id]).cuda()

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
            (text_encoded, mel, speaker_id, pitch_contour, rhythm))

    plot_mel_f0_alignment(x[2].data.cpu().numpy()[0],
                          mel_outputs_postnet.data.cpu().numpy()[0],
                          pitch_contour.data.cpu().numpy()[0, 0],
                          rhythm.data.cpu().numpy()[:, 0].T)
    plt.show()

    out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
    t0 = time.time()
    # wav = aukit.inv_mel_spectrogram()
    out_wav = infer_waveform_melgan(out_mel)
    print(time.time() - t0)
    aukit.play_audio(out_wav, sr=22050)

    t0 = time.time()
    with torch.no_grad():
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    out_wav = audio[0].data.cpu().numpy()
    print(time.time() - t0)
    aukit.play_audio(out_wav, sr=22050)


def singing_voice_v2():
    # Singing Voice from Music Score
    data = get_data_from_musicxml('data/sinsy/csongdb_f00002_000_en.musicxml', 132, convert_stress=True)
    panning = {'Soprano': [-60, -30], 'Alto': [-40, -10], 'Tenor': [30, 60], 'Bass': [10, 40]}
    n_speakers_per_part = 4
    frequency_scaling = 0.4
    n_seconds = 90
    audio_stereo = np.zeros((hparams.sampling_rate * n_seconds, 2), dtype=np.float32)
    for i, (part, v) in enumerate(data.items()):
        rhythm = data[part]['rhythm'].cuda()
        pitch_contour = data[part]['pitch_contour'].cuda()
        text_encoded = data[part]['text_encoded'].cuda()

        for k in range(n_speakers_per_part):
            pan = k
            # pan = np.random.randint(panning[part][0], panning[part][1])
            if any(x in part.lower() for x in ('soprano', 'alto', 'female')):
                speaker_id = torch.LongTensor([next(female_speakers)]).cuda()
            else:
                speaker_id = torch.LongTensor([next(male_speakers)]).cuda()
            print("{} MellotronID {} pan {}".format(part, speaker_id.item(), pan))

            with torch.no_grad():
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments_transfer = mellotron.inference_noattention(
                    (text_encoded, mel, speaker_id, pitch_contour * frequency_scaling, rhythm))

            plot_mel_f0_alignment(mel_outputs_postnet.data.cpu().numpy()[0],
                                  mel_outputs_postnet.data.cpu().numpy()[0],
                                  pitch_contour.data.cpu().numpy()[0, 0],
                                  rhythm.data.cpu().numpy()[:, 0].T)
            plt.show()

            out_mel = mel_outputs_postnet.data.cpu().numpy()[0]
            t0 = time.time()
            # wav = aukit.inv_mel_spectrogram()
            out_wav = infer_waveform_melgan(out_mel)
            print(time.time() - t0)

            aukit.save_wav(out_wav, "logs/musicxml_melgan_{}.wav".format(time.strftime("%Y%m%d-%H%M%S")), sr=22050)
            aukit.play_audio(out_wav, sr=22050)

            t0 = time.time()
            audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
            audio = audio.cpu().numpy()
            audio = panner(audio, pan)
            print(time.time() - t0)

            audio_stereo[:audio.shape[0]] += audio
            write("logs/{} {}.wav".format(part, speaker_id.item()), hparams.sampling_rate, audio)
            out_wav = audio

            aukit.play_audio(out_wav, sr=22050)

    # audio_stereo = audio_stereo / np.max(np.abs(audio_stereo))
    # write("audio_stereo.wav", hparams.sampling_rate, audio_stereo)
    # ipd.Audio([audio_stereo[:, 0], audio_stereo[:, 1]], rate=hparams.sampling_rate)


def griffinlim():
    # magnitudes = magnitudes.data
    # mel_output = torch.matmul(self.mel_basis, magnitudes)
    # mel_output = self.spectral_normalize(mel_output)
    mel = y[0]
    mel_de = dynamic_range_decompression(mel)[0]
    # torch.Size([80, 513]) torch.Size([4, 80, 402])

    mb_shape = valset.stft.mel_basis.shape
    mel_basis_ = torch.inverse(
        torch.cat((valset.stft.mel_basis, torch.rand(mb_shape[1] - mb_shape[0], mb_shape[1]) * 1e-9), 0))
    mel_de = torch.cat((mel_de.cpu(), torch.rand(mb_shape[1] - mb_shape[0], mel_de.shape[1]) * 1e-9), 0)
    magnitudes = torch.matmul(mel_basis_, mel_de)
    magnitudes = torch.unsqueeze(magnitudes, 0)
    wav_output = griffin_lim(magnitudes, stft_fn=valset.stft.stft_fn, n_iters=30)


def waveglow():
    waveglow_model = torch.load(r'../models/waveglow/waveglow_v5_model.pt', map_location='cpu')

    def waveglow_vocoder(mels):
        with torch.no_grad():
            wavs = waveglow_model.infer(mels, sigma=1.0)
        return wavs

    from mellotron.inference import transform_mel

    audio_ref, sr_ref = aukit.load_wav(audio, sr=None, with_sr=True)
    mel = transform_mel(audio_ref, stft=msyner.stft)
    spec_ref = mel

    wav_inputs = waveglow_vocoder(torch.from_numpy(spec_ref[None]))
    wav_ref = wav_inputs[0].cpu().numpy()


if __name__ == "__main__":
    print(__file__)
    # style_transfer()
    # singing_voice()
    style_transfer_v2()
    # singing_voice_v2()
