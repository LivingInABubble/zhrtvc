#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/21
"""
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .mel2wav.dataset import AudioDataset
from .mel2wav.interface import audio2mel, audio2mel_synthesizer, audio2mel_mellotron, get_default_device
from .mel2wav.modules import Generator, Discriminator
from .mel2wav.utils import save_sample

_device = get_default_device()


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练MelGAN声码器模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--data_path", type=str, default=r"../data/samples/metadata.csv",
                        help='metadata path')
    parser.add_argument("-o", "--save_path", type=str, default='../models/melgan/samples',
                        help=r"your model save dir")
    parser.add_argument("--load_path", type=str, default=None,
                        help=r"pretrained generator model path")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=4800)  # 8192

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=4)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--mode", type=str, default='mellotron')
    parser.add_argument("--ratios", type=str, default='5 5 4 2')  # '8 8 2 2'

    parser.add_argument("--cuda", type=str, default='0',
                        help='设置CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    return args


def train_melgan(args):
    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)

    metadata_dir = root.joinpath('metadata')
    metadata_dir.mkdir(exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(metadata_dir / "args.yml", "w") as f:
        yaml.dump(args.__dict__, f)
    with open(metadata_dir / "args.json", "w", encoding="utf8") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)

    eventdir = root / "events"
    eventdir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(eventdir))

    #######################
    # Load PyTorch Models #
    #######################
    ratios = [int(w) for w in args.ratios.split()]
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers, ratios=ratios).to(_device)
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    ).to(_device)
    # fft = Audio2Mel(n_mel_channels=args.n_mel_channels).to(_device)
    if args.mode == 'default':
        fft = audio2mel
    elif args.mode == 'synthesizer':
        fft = audio2mel_synthesizer
    elif args.mode == 'mellotron':
        fft = audio2mel_mellotron
    else:
        raise KeyError
    # print(netG)
    # print(netD)

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root))
        # optG.load_state_dict(torch.load(load_root / "optG.pt"))
        # netD.load_state_dict(torch.load(load_root / "netD.pt"))
        # optD.load_state_dict(torch.load(load_root / "optD.pt"))

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(args.data_path), args.seq_len, sampling_rate=args.sample_rate
    )
    test_set = AudioDataset(
        Path(args.data_path),  # test file
        args.sample_rate * 4,
        sampling_rate=args.sample_rate,
        augment=False,
    )

    # 保存训练数据
    with open(metadata_dir.joinpath('train.yml'), 'wt', encoding='utf8') as fout:
        yaml.dump([str(w.absolute()) for w in train_set.audio_files], fout,
                  default_flow_style=False, encoding='utf-8', allow_unicode=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.dataloader_num_workers,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    for i, x_t in enumerate(test_loader):
        x_t = x_t.to(_device)
        s_t = fft(x_t).detach()

        test_voc.append(s_t.to(_device))
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        oridir = root / "original"
        oridir.mkdir(exist_ok=True)
        save_sample(oridir / ("original_{}_{}.wav".format("test", i)), args.sample_rate, audio)
        writer.add_audio("original/{}/sample_{}.wav".format("test", i), audio, 0, sample_rate=args.sample_rate)
        mel_outputs = fft(x_t)
        writer.add_image(
            "original/{}/sample_{}.npy".format("test", i),
            plot_spectrogram_to_numpy(mel_outputs[0].data.cpu().numpy()),
            0, dataformats='HWC')
        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    step_begin = args.start_step
    look_steps = {step_begin + 10, step_begin + 100, step_begin + 1000, step_begin + 10000}
    steps = step_begin
    for epoch in range(1, args.epochs + 1):
        print("\nEpoch {} beginning. Current step: {}".format(epoch, steps))
        for iterno, x_t in enumerate(tqdm(train_loader, desc=f"Epoch-{epoch}", ncols=100)):
            # torch.Size([4, 1, 8192]) torch.Size([4, 80, 32])
            # 8192 = 32 x 256
            x_t = x_t.to(_device)
            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.to(_device))

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            #######################
            # Train Discriminator #
            #######################
            D_fake_det = netD(x_pred_t.to(_device).detach())
            D_real = netD(x_t.to(_device))

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.to(_device))

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            wt = D_weights * feat_weights
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + args.lambda_feat * loss_feat).backward()
            optG.step()

            ######################
            # Update tensorboard #
            ######################

            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])
            steps += 1
            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)

            if steps % args.save_interval == 0 or steps in look_steps:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, true_audio) in enumerate(zip(test_voc, test_audio)):
                        pred_audio_ = netG(voc)
                        pred_audio = pred_audio_.squeeze().cpu()
                        gendir = root / "generated"
                        gendir.mkdir(exist_ok=True)
                        save_sample(gendir / ("generated_step{}_{}.wav".format(steps, i)), args.sample_rate, pred_audio)
                        writer.add_audio(
                            "generated/step{}/sample_{}.wav".format(steps, i),
                            pred_audio,
                            epoch,
                            sample_rate=args.sample_rate,
                        )
                        # 查看频谱，直观了解生成语音的情况
                        mel_outputs = fft(pred_audio_.detach())
                        writer.add_image(
                            "generated/step{}/sample_{}.npy".format(steps, i),
                            plot_spectrogram_to_numpy(mel_outputs[0].data.cpu().numpy()),
                            epoch, dataformats='HWC')

                ptdir = root / "models"
                ptdir.mkdir(exist_ok=True)
                torch.save(netG.state_dict(), ptdir / "step{}_netG.pt".format(steps))
                torch.save(optG.state_dict(), ptdir / "step{}_optG.pt".format(steps))

                torch.save(netD.state_dict(), ptdir / "step{}_netD.pt".format(steps))
                torch.save(optD.state_dict(), ptdir / "step{}_optD.pt".format(steps))

                if (np.asarray(costs).mean(0)[-1] < best_mel_reconst) or (steps % (args.save_interval * 10) == 0):
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD, ptdir / "best_step{}_netD.pt".format(steps))
                    torch.save(netG, ptdir / "best_step{}_netG.pt".format(steps))

                # print("\nTook %5.4fs to generate samples" % (time.time() - st))
                # print("-" * 100)

            if steps % args.log_interval == 0 or steps in look_steps:
                print(
                    "\nEpoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


if __name__ == "__main__":
    args = parse_args()
    train_melgan(args)
