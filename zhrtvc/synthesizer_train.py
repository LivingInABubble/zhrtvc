import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(Path(__file__).stem)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def prepare_run(args):
    from synthesizer.hparams import hparams
    from synthesizer import infolog

    modified_hp = hparams.parse(args.hparams)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_log_level)
    run_name = args.name
    log_dir = os.path.join(args.models_dir, "logs-{}".format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, Path(__file__).stem + '.log'), run_name, args.slack_url)

    return log_dir, modified_hp


def main():
    from synthesizer.train import tacotron_train
    from utils.argutils import print_args

    parser = ArgumentParser(description="训练语音合成器模型。", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default='synz', help="Name of the run and of the logging directory.")
    parser.add_argument("--synthesizer_root", type=str, default=r'../data/SV2TTS/synthesizer',
                        help="Path to the synthesizer training data that contains the audios and the train.txt file. "
                             "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-m", "--models_dir", type=str, default="../models/synthesizer/saved_models/",
                        help="Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("--mode", default="synthesis", help="mode for synthesis of tacotron after training")
    parser.add_argument("--GTA", default="True", help="Ground truth aligned synthesis, defaults to True, "
                                                      "only considered in Tacotron synthesis mode")
    parser.add_argument("--restore", type=bool, default=True, help="Set this to False to do a fresh training")
    parser.add_argument("--summary_interval", type=int, default=100, help="Steps between running summary ops")
    parser.add_argument("--embedding_interval", type=int, default=100,
                        help="Steps between updating embeddings projection visualization")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,  # Was 5000
                        help="Steps between writing checkpoints")
    parser.add_argument("--eval_interval", type=int, default=100,  # Was 10000
                        help="Steps between eval on test data")
    parser.add_argument("--tacotron_train_steps", type=int, default=500000,  # Was 100000
                        help="total number of tacotron training steps")
    parser.add_argument("--tf_log_level", type=int, default=1, help="Tensorflow C++ log level.")
    parser.add_argument("--slack_url", default=None, help="slack webhook notification destination link")
    parser.add_argument("--hparams", default="",
                        help="Hyperparameter overrides as a json string, for example: '\"key1\":123,\"key2\":true'")
    args = parser.parse_args()
    print_args(args, parser)

    log_dir, hparams = prepare_run(args)
    tacotron_train(args, log_dir, hparams)


if __name__ == '__main__':
    main()
