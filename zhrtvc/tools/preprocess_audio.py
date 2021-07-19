import collections as clt
import os
from pathlib import Path

import aukit
import numpy as np
from cycler import cycle
from tqdm import tqdm

_sr = 16000


def remove_noise_and_silence(wav, sr=_sr):
    """
    remove noise and silence
    :param wav:
    :param sr:
    :return:
    """
    out = aukit.remove_noise(wav, sr=sr)
    out = out / np.max(np.abs(out))
    out = aukit.remove_silence_wave(out, sr=sr, keep_silence_len=50, min_silence_len=100, silence_thresh=-32)
    out = aukit.strip_silence_wave(out, sr=sr, keep_silence_len=20, min_silence_len=40, silence_thresh=-32)
    return out


def joint_audio_and_text(pairs):
    """
    get joint audio and text
    :param pairs: (wav, text)
    :param sr:
    :return:
    """
    wavs, texts = [], []
    for w, t in pairs:
        wavs.append(w)
        texts.append(t)
    out_wav = np.concatenate(tuple(wavs), axis=0)
    out_text = "。".join(texts)
    return out_wav, out_text


def choice_pairs(pairs, n_choice=100):
    """
    choice audios and texts to joint
    :param pairs: (audio_path, text)
    :param sr:
    :param n_choice:
    :return:
    """
    durs = [os.path.getsize(str(w)) / 3.2 for w, t in pairs]  # just sr=16k
    pairs = list(zip(pairs, durs))
    sidx = 0
    choices = cycle(range(5, 1, -1))
    cnt = 0
    cnt_shuffle = 0
    for n in choices:
        if sidx >= len(pairs):
            if cnt_shuffle >= n_choice:  # don't work forever
                print("Just choice <{}> joint audios for the speaker!".format(cnt))
                break
            cnt_shuffle += 1
            np.random.shuffle(pairs)
            sidx = 0
        eidx = sidx + n
        dur_joi = np.sum([d for _, d in pairs[sidx: eidx]])
        if dur_joi < 12000:  # 12000ms
            if cnt >= n_choice:
                break
            yield [p for p, d in pairs[sidx: eidx]]
            cnt += 1
            sidx = eidx
        elif n == 2:
            sidx += 1


def load_pairs(fpath):
    """
    load path and text pairs
    :param fpath:
    :return:
    """
    curdir = Path(fpath).parent
    spk_pairs_dt = clt.defaultdict(list)
    for line in open(fpath, encoding="utf8"):
        index, text = line.strip().split("\t")
        spk = Path(index).parent.name
        spk_pairs_dt[spk].append((curdir.joinpath(index), text))
    for spk, pairs in spk_pairs_dt.items():
        yield (spk, pairs)


def run_joint(fpath, sr=_sr, outdir=Path("")):
    """
    run joint
    :param fpath:
    :param sr:
    :param outdir:
    :return:
    """
    curdir = Path(fpath).parent
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir.joinpath("metadata.csv"), "wt", encoding="utf8") as fout:
        load_pair = load_pairs(fpath)
        for spk, ptpairs_raw in tqdm(load_pair, desc="speaker", ncols=100):
            gen_pair = choice_pairs(ptpairs_raw, n_choice=100)
            for num, ptpairs_joint in enumerate(tqdm(gen_pair, desc="choice", ncols=100), 1):
                wtpairs_joint = [(aukit.load_wav(p, sr=sr), t) for p, t in ptpairs_joint]
                wav, text = joint_audio_and_text(wtpairs_joint)
                parts = list(Path(ptpairs_joint[0][0]).relative_to(curdir).parts)[:-1]
                parts.append("{}_{:06d}.wav".format(spk, num))
                outname = "/".join(parts)
                outpath = outdir.joinpath(outname)
                outpath.parent.mkdir(exist_ok=True, parents=True)
                aukit.save_wav(wav, sr=sr, path=outpath)
                fout.write("{}\t{}\n".format(outname, text))


if __name__ == "__main__":
    metadata_path = "../data/samples/metadata.csv"
    sr = 16000
    outdir = "../data/samples_joint"
    run_joint(metadata_path, sr=sr, outdir=outdir)
