from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from toolbox.core import Toolbox
from utils.argutils import print_args

# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(Path(__file__).stem)


def main():
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    parser = ArgumentParser(description="可视化分析的Demo。", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datasets_root", type=Path, default=r"../data/samples",
                        help="Path to the directory containing your datasets.")
    parser.add_argument("-e", "--enc_models_dir", type=Path, default="../models/encoder/saved_models",
                        help="Directory containing saved encoder models")
    parser.add_argument("-s", "--syn_models_dir", type=Path, default="../models/synthesizer/saved_models",
                        help="Directory containing saved synthesizer models")
    parser.add_argument("-v", "--voc_models_dir", type=Path, default="../models/vocoder/saved_models",
                        help="Directory containing saved vocoder models")
    parser.add_argument("-t", "--toolbox_files_dir", type=Path, default="../models/toolbox/saved_files",
                        help="Directory containing saved toolbox files")
    parser.add_argument("--low_mem", action="store_true",
                        help="If True, the memory used by the synthesizer will be freed after each use. "
                             "Adds large overhead but allows to save some GPU memory for lower-end GPUs.")
    args = parser.parse_args()

    # Launch the toolbox
    print_args(args, parser)
    Toolbox(**vars(args))


if __name__ == '__main__':
    main()
