from pathlib import Path
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesizer.preprocess import create_embeddings
from utils.argutils import print_args
from synthesizer.hparams import hparams
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="把语音信号转为语音表示向量。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--synthesizer_root", type=Path, default=Path(r'../data/SV2TTS/synthesizer'), help= \
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default=r"../models/encoder/saved_models/ge2e_pretrained.pt", help= \
                            "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=0, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    parser.add_argument("--hparams", type=str, default="", help= \
        "Hyperparameter overrides as a json string, for example: '\"key1\":123,\"key2\":true'")
    args = parser.parse_args()

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)
    create_embeddings(**vars(args))
