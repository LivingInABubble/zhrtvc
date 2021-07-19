import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from synthesizer.preprocess import preprocess_user
    from synthesizer.hparams import hparams
    from utils.argutils import print_args

    parser = argparse.ArgumentParser(
        description="把语音信号转为频谱等模型训练需要的数据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--datasets_root", type=Path, default=Path(r'../data'),
                        help="Path to the directory containing your datasets.")
    parser.add_argument("--datasets", type=str, default="zhujingxi_voice",
                        help="Path to the directory containing your datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS,
                        help="Path to the output directory that will contain the mel spectrograms, "
                             "the audios and the embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=0, help="Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", type=bool, default=True,
                        help="Whether to overwrite existing files with the same name. "
                             "Useful if the preprocessing was interrupted.")
    parser.add_argument("--hparams", type=str, default="",
                        help="Hyperparameter overrides as a json string, for example: '\"key1\":123,\"key2\":true'")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)
    preprocess_user(**vars(args))


if __name__ == '__main__':
    main()
