#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/4/13
"""
"""
from pathlib import Path

import librosa
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def gen_filelists():
    indir = Path(r"E:\data\librispeech\LibriSpeech\test-clean\LibriSpeech\test-clean")
    # 61-70968-0001: speaker-book-id
    # 61-70968-0001 GIVE NOT SO EARNEST A MIND TO THESE MUMMERIES CHILD
    outs = []
    for fpath in indir.glob("**/*.txt"):
        for line in open(fpath):
            idx, *ws = line.strip().split()
            outpath = str(fpath.parent.joinpath(f"{idx}.flac")).replace("\\", "/")
            outtext = (" ".join(ws)).lower()
            outspeaker = idx.split("-")[0]
            outs.append([outpath, outtext, outspeaker])

    np.random.shuffle(outs)

    outpath = r"E:\data\librispeech\LibriSpeech\test-clean\lstc_total_filelist.txt"
    with open(outpath, "wt", encoding="utf8") as fout:
        for line in tqdm(outs):
            if 10 < len(line[1]) < 50:
                wav, sr = librosa.load(line[0])
                wav = (wav * (2 ** 15)).astype(int)
                if max(wav) >= 2 ** 15:
                    print(line)
                    continue
                fout.write(("|".join(line)) + "\n")


def read(fpath):
    wav, sr = librosa.load(fpath)
    out = wav * (2 ** 15 - 1)
    return sr, out.astype(int)


def load_flac():
    import librosa
    inpath = r"E:/data/librispeech/LibriSpeech/test-clean/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
    inpath = r"E:\data\record\kdd\wavs\kdd-4253.wav"
    inpath = r"E:/data/librispeech/LibriSpeech/test-clean/LibriSpeech/test-clean/2094/142345/2094-142345-0025.flac"
    # sr, wav = wavfile.read(inpath)
    wav2, sr = librosa.load(inpath)
    print(max(abs(wav2)))


def run_aliaudio():
    indir = Path(r'F:\data\alijuzi_tar')
    txtpath = indir.joinpath('biaobei_juzi_ssml.txt')
    # spk_lst = [w.stem for w in sorted(indir.glob('*.tar'))]
    spk_lst = 'Aibao Aicheng Aida Aijia Aijing Aimei Aina Aiqi Aitong Aiwei Aixia Aiya Aiyu Aiyue Sijia Siqi Siyue Xiaobei Xiaomei Sicheng'.split()

    text_lst = open(txtpath, encoding='utf8').readlines()
    outpath = indir.joinpath('metadata.csv')
    with open(outpath, 'wt', encoding='utf8') as fout:
        for spk in tqdm(spk_lst):
            for line in text_lst:
                idx, text = line.strip().split('\t')
                audio_path = f'{spk}/{idx}.wav'
                out = f'{audio_path}\t{text}\t{spk}\n'
                fout.write(out)


def run_umap():
    import umap
    from sklearn.manifold import TSNE
    from sklearn import manifold
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import FastICA

    txt_fpath = r'../../data/SV2TTS/mellotron/linear/train.txt'
    npy_dir = Path(txt_fpath).parent.joinpath('npy')
    ids = []
    with open(txt_fpath, encoding='utf8') as fin:
        for line in fin:
            embed_fpath = npy_dir.joinpath(line.split('\t')[0], 'embed.npy')
            ids.append(embed_fpath)

    data = []
    data_train = []
    for num, fpath in enumerate(ids):
        vec = np.load(fpath)
        if not (150 <= num < 250):
            data.append(vec)
        else:
            data_train.append(vec)
    data = np.array(data)
    data_train = np.array(data_train)
    print(data.shape, data_train.shape)

    n_dim = 3
    umap_data_3 = umap.UMAP(n_components=n_dim, n_neighbors=5, min_dist=0.9).fit_transform(data)
    tsne_data_3 = TSNE(n_components=3, n_iter=300).fit_transform(data)
    isomap_data_3 = manifold.Isomap(n_components=n_dim, n_neighbors=5, n_jobs=-1).fit_transform(data)
    pca_data_3 = PCA(n_components=n_dim).fit_transform(data)
    svd_data_3 = TruncatedSVD(n_components=n_dim, random_state=42).fit_transform(data)
    ica_data_3 = FastICA(n_components=n_dim, random_state=12).fit_transform(data)
    top_data_3 = data[:, ::data.shape[1] // n_dim]
    top_data_16 = data[:, ::256 // n_dim]

    pca = PCA(n_components=n_dim, random_state=42)
    pca_model = pca.fit(data_train)
    top_data_16 = pca_model.transform(data)

    # n_neighbors确定使用的相邻点的数量
    # min_dist控制允许嵌入的紧密程度。值越大，嵌入点的分布越均匀
    # 让我们可视化一下这个变换：

    # umap_model = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2)
    # out_model = umap_model.fit(data)
    # umap_data = out_model.transform(data)
    # umap_data = umap_data_3
    # plt.figure(figsize=(12, 8))
    # plt.title('Decomposition using UMAP')
    # plt.scatter(umap_data[:, 0], umap_data[:, 1])
    # plt.scatter(umap_data[:, 1], umap_data[:, 2])
    # plt.scatter(umap_data[:, 2], umap_data[:, 0])
    # plt.show()

    from sklearn.metrics.pairwise import cosine_similarity

    plt.subplot('331')
    plt.imshow(cosine_similarity(data, data))
    plt.subplot('332')
    plt.imshow(cosine_similarity(top_data_3, top_data_3))
    plt.subplot('333')
    plt.imshow(cosine_similarity(top_data_16, top_data_16))
    plt.subplot('334')
    plt.imshow(cosine_similarity(umap_data_3, umap_data_3))
    plt.subplot('335')
    plt.imshow(cosine_similarity(tsne_data_3, tsne_data_3))
    plt.subplot('336')
    plt.imshow(cosine_similarity(isomap_data_3, isomap_data_3))
    plt.subplot('337')
    plt.imshow(cosine_similarity(pca_data_3, pca_data_3))
    plt.subplot('338')
    plt.imshow(cosine_similarity(svd_data_3, svd_data_3))
    plt.subplot('339')
    plt.imshow(cosine_similarity(ica_data_3, ica_data_3))

    plt.show()


def combine_train_txt(*args, outpath):
    outs = []
    for inpath in args:
        tmp = '/'.join(Path(inpath).parent.relative_to(Path(outpath).parent).parts)
        for line in open(inpath, encoding='utf8'):
            outs.append(f'{tmp}/{line}')

    with open(outpath, 'wt', encoding='utf8') as fout:
        for line in outs:
            fout.write(line)


if __name__ == "__main__":
    print(__file__)
    # gen_filelists()
    # load_flac()
    # run_aliaudio()
    # run_umap()
    combine_train_txt(r'F:\github\zhrtvc\data\samples\metadata.csv',
                      r'F:\github\zhrtvc\data\samples_ssml\metadata.csv',
                      outpath=r'F:\github\zhrtvc\data\metadata.csv')
