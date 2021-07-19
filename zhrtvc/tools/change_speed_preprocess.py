#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/25
"""
"""
import json
import os
import re
import shutil
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import aukit
import numpy as np
import pydub
from tqdm import tqdm

from melgan.inference import wav2mel, infer_waveform_melgan, melgan_hparams, Dict2Obj

_sr = 16000

_melgan_load_path = r"../vocoder/saved_models/melgan/multi_speaker.pt"

_speeds = [0.7, 0.85, 1, 1.15, 1.3]
_speeds_str = "0.7 0.85 1 1.15 1.3".split()


class DataJoint():
    def __init__(self):
        pass


def gen_audio(meta_path: Path, speaker: str):
    itdt = {}
    with open(meta_path, encoding="utf8") as fin:
        for line in fin:
            idx, text = line.strip().split("\t")
            itdt[idx] = text

    spkpath = meta_path.parent.joinpath("speed/{}_{}_path.txt".format(speaker, meta_path.parent.name))
    stpdt = get_map_path(spkpath)

    t_yx = []
    for t in itdt.keys():
        for s in _speeds_str:
            key = "{}/{}".format(s, t)
            if key not in stpdt:
                break
        else:
            t_yx.append(t)
    print(dict(t_yx=len(t_yx)))

    t_ch = choose_text(t_yx, num_per=500, range_args=(2, 10))
    s_ch = []
    for w in t_ch:
        s = assign_speed(w)
        s_ch.append(s)

    kw_lst = []
    for num, (text, speed) in enumerate(zip(t_ch, s_ch), 1):
        outidx = "joint_{:06d}".format(num)
        keys = ["{}/{}".format(s, t) for t, s in zip(text, speed)]
        inpaths = [str(stpdt[key]) for key in keys]
        outpath = str(
            meta_path.parent.joinpath("speed/{}_{}_joint/{}.mp3".format(speaker, meta_path.parent.name, outidx)))
        outtext = " | ".join([itdt[t] for t in text])
        outinfo = " | ".join(["{}/{}".format(meta_path.parent.name, key) for key in keys])
        kw = dict(inpaths=inpaths, outpath=outpath, outtext=outtext, outinfo=outinfo, outidx=outidx)
        kw_lst.append(kw)

    outkwargs_path = meta_path.parent.joinpath("speed/{}_{}_joint_kwargs.json".format(speaker, meta_path.parent.name))
    outtext_path = meta_path.parent.joinpath("speed/{}_{}_joint_text.txt".format(speaker, meta_path.parent.name))
    outinfo_path = meta_path.parent.joinpath("speed/{}_{}_joint_info.txt".format(speaker, meta_path.parent.name))
    if outkwargs_path.exists():
        print(outkwargs_path)
        flag = input(
            "The path exists. Please check.\nOption 1: run new kwargs\nOption 2: run old kwargs\nOther: break\nSelect option:")
        if flag == "1":
            with open(outkwargs_path, "wt", encoding="utf8") as foutkwargs:
                json.dump(kw_lst, foutkwargs, indent=4, ensure_ascii=False)

            with open(outtext_path, "wt", encoding="utf8") as fouttext:
                for kw in tqdm(kw_lst):
                    fouttext.write("{}\t{}\n".format(kw["outidx"], kw["outtext"]))

            with open(outinfo_path, "wt", encoding="utf8") as foutinfo:
                for kw in tqdm(kw_lst):
                    foutinfo.write("{}\t{}\n".format(kw["outidx"], kw["outinfo"]))

        elif flag == "2":
            print("run <{}>".format(outkwargs_path))
            kw_lst = json.load(open(outkwargs_path, "rt", encoding="utf8"))
            kw_lst = [{**kw, **{"skip_exist": True}} for kw in kw_lst]
        else:
            return

    run_many(kw_lst, func=joint_audio_one, n_processes=8)


def get_path_speaker(indir: Path, speaker: str):
    outs = []
    for fpath in sorted(indir.glob("*/{}/*.mp3".format(speaker))):
        outs.append(fpath)

    outpath = indir.joinpath("{}_{}_path.txt".format(speaker, indir.parent.name))
    with open(outpath, "wt", encoding="utf8") as fout:
        for fpath in outs:
            fout.write("{}\n".format(str(fpath).replace("\\", "/")))


def get_map_path(inpath):
    # E:/data/aliaudio/aliduanyu/speed/mp3_speed0.7/aijia/000001_03_hua2.mp3
    _match_re = re.compile(r"^E:/.+?/speed/.+?speed(.+?)/(.+?)/(.+)\.mp3$")
    paths = [w.strip() for w in open(inpath, encoding="utf8")]
    outdt = {}
    for path in paths:
        m = _match_re.search(path)
        if m:
            key = "{}/{}".format(m.group(1), m.group(3))
            outdt[key] = path
        else:
            print(path)
    return outdt


def choose_text(src, num_per=500, range_args=(2, 10)):
    outs = []
    for num in range(*range_args):
        n_total = 0
        while n_total < num_per:
            np.random.shuffle(src)
            n_cur = len(src) // num
            n_cur = min(num_per - n_total, n_cur)
            out_cur = np.reshape(src[:n_cur * num], (n_cur, num))
            outs.extend(out_cur)
            n_total += n_cur
    return outs


def assign_speed(src):
    out = np.random.choice(_speeds_str, len(src))
    return out


def joint_audio_one(kwargs: dict):
    inpaths = kwargs.get("inpaths")
    outpath = Path(kwargs.get("outpath"))
    skip_exist = kwargs.get("skip_exist")
    if outpath.exists() and skip_exist:
        return
    outpath.parent.mkdir(exist_ok=True, parents=True)

    snds = []
    for inpath in inpaths:
        snd = pydub.AudioSegment.from_mp3(inpath)
        snds.append(snd)
    out = pydub.AudioSegment.silent(duration=200, frame_rate=16000)
    for snd in snds:
        out = out.append(snd, crossfade=200)

    out.export(outpath, format="mp3")
    return kwargs


def change_speed_one(kwargs: dict):
    inpath = kwargs.get("inpath")
    outpath = kwargs.get("outpath")
    rate = kwargs.get("rate")
    if Path(outpath).exists() and os.path.getsize(outpath) > 8000:
        return
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    hp = Dict2Obj()
    hp.update(melgan_hparams)
    hp.update({"hop_size": int(melgan_hparams["hop_size"] * rate)})

    try:
        wav = aukit.load_wav(inpath, sr=_sr)
        mel = wav2mel(wav, hparams=hp)
        out = infer_waveform_melgan(mel, load_path=_melgan_load_path)
        aukit.save_wav(out, outpath, sr=_sr)
    except Exception as e:
        print(e)
        print(kwargs)
    return kwargs


def run_many(kwargs_list, func, n_processes):
    if n_processes == 0:
        for kw in tqdm(kwargs_list, str(n_processes), unit="it"):
            func(kwargs=kw)
    else:
        pfunc = partial(func)
        job = Pool(n_processes).imap(pfunc, kwargs_list)

        for kw in tqdm(job, str(n_processes), len(kwargs_list), unit="it"):
            pass


def process_aliaudio(indir: Path, n_processes=8):
    oov_speakers = set("Aitong Aiwei Ninger Ruilin Ruoxi Sitong Xiaobei Xiaoyun Yina".split())
    choice_speakers = set("Aibao Aicheng Aijia Aina".split())
    kwargs_list = []
    for rate in [1]:  # [0.7, 0.85, 1.15, 1.3]:
        outdir = indir.parent.joinpath("{}_speed{}".format(indir.stem, rate))
        for fpath in indir.glob("**/*.mp3"):
            if fpath.parent.stem not in choice_speakers:
                continue
            outpath = outdir.joinpath("/".join(fpath.relative_to(indir).parts))
            kw = dict(inpath=fpath, outpath=outpath, rate=rate)
            kwargs_list.append(kw)

    run_many(kwargs_list, func=change_speed_one, n_processes=n_processes)


def mp32wav_one(kwargs: dict):
    inpath = str(kwargs.get("inpath")).replace("/", "\\")
    outpath = str(kwargs.get("outpath")).replace("/", "\\")
    if Path(outpath).exists():
        return
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    try:
        aus = pydub.AudioSegment.from_mp3(inpath)
        aus.export(outpath, format="wav")
    except Exception as e:
        print(e)
        print(kwargs)


def wav2mp3_one(kwargs: dict):
    inpath = str(kwargs.get("inpath")).replace("/", "\\")
    outpath = str(kwargs.get("outpath")).replace("/", "\\")
    if Path(outpath).exists() and os.path.getsize(outpath) > 1600:
        return
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    try:
        if os.path.getsize(inpath) < 8000:
            shutil.copyfile(inpath, outpath)
        else:
            aus = pydub.AudioSegment.from_wav(inpath)
            aus.export(outpath, format="mp3")
    except Exception as e:
        print(e)
        print(kwargs)


def convert_many(indir: Path, func=None, n_processes=8):
    choice_speakers = set("Aibao Aicheng Aijia Aina".split())
    kwargs_list = []
    outdir = indir.parent.joinpath("{}_mp3".format(indir.stem))
    for fpath in indir.glob("**/*.mp3"):
        # if fpath.parent.stem not in choice_speakers:
        #     continue
        outpath = outdir.joinpath("/".join(fpath.parent.relative_to(indir).parts), fpath.stem + ".mp3")
        kw = dict(inpath=fpath, outpath=outpath)
        kwargs_list.append(kw)
    run_many(kwargs_list, func=func, n_processes=n_processes)


if __name__ == "__main__":
    print(__file__)
    # indir = Path(r"E:\data\aliaudio\alipinyin\mp3")
    # process_aliaudio(indir, n_processes=0)

    # indir = Path(r"E:\data\aliaudio\alijuzi\mp3_speed1")
    # convert_many(indir, func=wav2mp3_one, n_processes=0)

    # inpaths = []
    # for fpath in Path(r"E:\data\aliaudio\temp").glob("*.mp3"):
    #     inpaths.append(fpath)
    # outpath = r"E:\data\aliaudio\temp\joint.100-100.wav"
    # kw = dict(inpaths=inpaths, outpath=outpath)
    # joint_audio_one(kw)
    # aukit.play_audio(outpath)

    # outs = choose_audio(list(range(10)))
    # print(len(outs))

    # indir = Path(r"E:\data\aliaudio\alijuzi\speed")
    # for spk in "Aibao Aicheng Aijia Aina".split():
    #     get_path_speaker(indir, spk)

    # inpath = r"E:\data\aliaudio\aliduanyu\speed\Aibao_aliduanyu_path.txt"
    # outdt = get_map_path(inpath)
    # print(outdt)

    meta_path = Path(r"E:\data\aliaudio\aliduanyu\meta.csv")
    speaker = "Aina"
    gen_audio(meta_path=meta_path, speaker=speaker)
