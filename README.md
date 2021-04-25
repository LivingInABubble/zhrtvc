![zhrtvc](data/files/zhrtvc.png "zhrtvc")

# zhrtvc
Chinese Real Time Voice Cloning

tips: 中文或汉语的语言缩写简称是**zh**。

关注【**啊啦嘻哈**】微信公众号，回复一个字【**听**】，小萝莉有话对你说哦^v^

![啊啦嘻哈](data/files/alaxiha.jpg "啊啦嘻哈")


### 语音合成工具箱
+ 便携使用的语音合成工具箱[ttskit](https://github.com/KuangDD/ttskit)
+ ttskit基于本项目[zhrtvc](https://github.com/KuangDD/zhrtvc)，侧重于语音合成的使用。
+ 快速使用：
```python
# 安装：pip install ttskit

import ttskit

# 合成语音
ttskit.tts('这是个样例', audio='1')
```

### 版本

v1.4.23

使用说明和注意事项详见[README](zhrtvc/README.md)

+ **注意事项**: 

    - 这个说明是新版**GMW版本**的语音克隆框架的说明，使用ge2e(encoder)-mellotron-waveglow的模块（简称GMW），运行更简单，效果更稳定和合成语音更加优质。

    - 基于项目Real-Time-Voice-Cloning改造为中文支持的版本ESV版本的说明见[README-ESV](README-ESV.md)，该版本用encoder-synthesizer-vocoder的模块（简称ESV），运行比较复杂。

    - 需要进入zhrtvc项目的代码子目录【[zhrtvc](zhrtvc)】运行代码。

    - zhrtvc项目默认参数设置是适用于data目录中的样本数据，仅用于跑通整个流程。

    - 推荐使用mellotron的语音合成器和waveglow的声码器，mellotron设置多种模式适应多种任务使用。


+ **中文语料**

中文语音语料[zhvoice](https://github.com/KuangDD/zhvoice)，语音更加清晰自然，包含8个开源数据集，3200个说话人，900小时语音，1300万字。

+ **中文模型**

扫描上面的二维码，关注**【啊啦嘻哈】微信公众号**，回复：**中文语音克隆模型走起**，获取百度网盘的模型文件。

+ **合成样例**

[合成语音样例的目录](data/files/examples)

### 目录介绍

#### zhrtvc

代码相关的说明详见zhrtvc目录下的[readme](zhrtvc/README.md)文件。

#### models

预训练的模型在百度网盘下载，下载后解压，替换models文件夹即可。

#### data
语料样例，包括语音和文本对齐语料。

**注意：** 

该语料样例用于测试跑通模型，数据量太少，不可能使得模型收敛，即不会训练出可用模型。

在测试跑通模型情况下，处理自己的数据为语料样例的格式，用自己的数据训练模型即可。


### 学习交流

【AI解决方案交流群】QQ群：925294583

点击链接加入群聊：https://jq.qq.com/?_wv=1027&k=wlQzvT0N


### Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. Feel free to check [my thesis](https://matheo.uliege.be/handle/2268.2/6801) if you're curious or if you're looking for info I haven't documented yet (don't hesitate to make an issue for that too). Mostly I would recommend giving a quick look to the figures beyond the introduction.

SV2TTS is a three-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model trained to generalize to new voices.
