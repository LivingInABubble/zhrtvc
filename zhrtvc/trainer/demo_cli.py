from pathlib import Path
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from synthesizer import hparams
from synthesizer.utils import audio
from encoder import inference as encoder
# from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import time
import torch
import sys
import shutil
import json

import aukit
from utils.texthelper import xinqing_texts

example_texts = xinqing_texts

sample_dir = Path(r"../data/files")
reference_paths = [w for w in sorted(sample_dir.glob('*.wav'))]

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        description="命令行执行的Demo。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="../models/encoder/saved_models/ge2e_pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="../models/synthesizer/saved_models/logs-syne/checkpoints",  # pretrained
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="../models/vocoder/saved_models/melgan/melgan_multi_speaker.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-o", "--out_dir", type=Path,
                        default="../data/outs",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)
    # if not args.no_sound:
    #     import sounddevice as sd

    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath, device='cpu')

    # 从模型目录导入hparams
    hp_path = args.syn_model_dir.parent.joinpath("metas", "hparams.json")    # load from trained models
    if hp_path.exists():
        hparams = aukit.Dict2Obj(json.load(open(hp_path, encoding="utf8")))
        print('hparams:')
        print(json.dumps({k:v for k, v in hparams.items()}, ensure_ascii=False, indent=4))
    else:
        hparams = None
        print('hparams:', hparams)

    synthesizer = Synthesizer(args.syn_model_dir, low_mem=args.low_mem, hparams=hparams)

    # vocoder.load_model(args.voc_model_fpath)

    ## Run a test
    print("Testing your configuration with small inputs.")
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["你好", "欢迎使用语音克隆工具"]
    print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)

    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None

    generated_wav = audio.inv_melspectrogram(mel, hparams=audio.melgan_hparams)
    print("All test passed! You can now synthesize speech.\n\n")

    print("Interactive generation loop")
    num_generated = 0
    args.out_dir.mkdir(exist_ok=True, parents=True)
    while True:
        try:
            # Get the reference audio filepath
            message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                      "wav, m4a, flac, ...):\n"
            ref = input(message)
            in_fpath = Path(ref.replace("\"", "").replace("\'", ""))
            if not in_fpath.is_file():
                in_fpath = np.random.choice(reference_paths)
            print('Reference audio: {}'.format(in_fpath))
            # - Directly load from the filepath:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            print("Loaded file succesfully")

            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")

            ## Generating the spectrogram
            text = input("Write a sentence (+-20 words) to be synthesized:\n")

            if not text.strip():
                text = np.random.choice(example_texts)

            print('Text: {}'.format(text))
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]

            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")

            ## Generating the waveform
            print("Synthesizing the waveform:")

            generated_wav = audio.inv_melspectrogram(spec, hparams=audio.melgan_hparams)
            # generated_wav = synthesizer.griffin_lim(spec, hparams=synthesizer.hparams)
            # generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Play the audio (non-blocking)
            # if not args.no_sound:
            #     sd.stop()
            #     sd.play(generated_wav, synthesizer.sample_rate)
            #     sd.wait()

            # Save it on the disk
            cur_time = time.strftime('%Y%m%d_%H%M%S')
            fpath = args.out_dir.joinpath("demo_out_{}.wav".format(cur_time))
            # librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
            audio.save_wav(generated_wav, fpath, synthesizer.sample_rate)  # save

            ref_path = args.out_dir.joinpath("demo_ref_{}.mp3".format(cur_time))
            shutil.copyfile(in_fpath, ref_path)

            txt_path = args.out_dir.joinpath("info_dict.txt".format(cur_time))
            with open(txt_path, 'at', encoding='utf8') as fout:
                dt = dict(text=text, audio_path=str(fpath), reference_path=str(in_fpath), time=cur_time)
                out = json.dumps(dt, ensure_ascii=False)
                fout.write('{}\n'.format(out))

            num_generated += 1
            print("\nSaved output as %s\n\n" % fpath)
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
