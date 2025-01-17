# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import sys

import torch
from scipy.io.wavfile import write

sys.path.append('waveglow')

_model = None
_denoiser = None


def load_waveglow_model(model_path, device=None):
    """
    导入训练模型得到的checkpoint模型文件。
    """
    from .denoiser import Denoiser

    global _model
    global _denoiser
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if _model is None:
        _model = torch.load(model_path, map_location=device)['model']
        _model = _model.remove_weightnorm(_model)
        _model.to(device).eval()
        _denoiser = Denoiser(_model).to(device)


def load_waveglow_torch(model_path, device=None):
    """
    用torch.load直接导入模型文件，不需要导入模型代码。
    """
    from .denoiser import Denoiser

    global _model
    global _denoiser
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = torch.load(model_path, map_location=device)
    _denoiser = Denoiser(_model).to(device)

    return _model


def is_loaded():
    """
    判断模型是否已经导入。
    """
    global _model
    return _model is not None


def generate_wave(mel, **kwargs):
    """
    用声码器模型把mel频谱转为音频信号。
    """
    global _model
    global _denoiser
    sigma = kwargs.get('sigma', 1.0)
    denoiser_strength = kwargs.get('denoiser_strength', 0)
    if not is_loaded():
        load_waveglow_model(**kwargs)
    mel = torch.autograd.Variable(mel)

    with torch.no_grad():
        wav = _model.infer(mel, sigma=sigma)
        if denoiser_strength > 0:
            wav = _denoiser(wav, denoiser_strength)
            wav = wav.squeeze(0)

        return wav


def main(mel_files, waveglow_path, sigma, output_dir, sampling_rate, is_fp16, denoiser_strength):
    from .denoiser import Denoiser
    from .mel2samp import files_to_list, MAX_WAV_VALUE

    mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    # if is_fp16:
    #     from apex import amp
    #     waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    for i, file_path in enumerate(mel_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        mel = torch.load(file_path)
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)

            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}_synthesis.wav".format(file_name))
        write(audio_path, sampling_rate, audio)
        print(audio_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path', help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir, args.sampling_rate, args.is_fp16,
         args.denoiser_strength)
