import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from time import perf_counter as timer

import numpy as np
from aukit import remove_noise, Dict2Obj
from aukit.audio_griffinlim import save_wav
from aukit.audio_normalizer import trim_long_silences

from encoder import inference as encoder
from encoder.audio import preprocess_wav
from melgan import inference as vocoder_melgan
from synthesizer.inference import Synthesizer
from toolbox.ui import UI
from toolbox.utterance import Utterance
from vocoder import inference as vocoder
from .sentence import xinqing_texts

# Use this directory structure for your datasets, or modify it to fit your needs

filename_formatter_re = re.compile(r'[\s\\/:*?"<>|\']+')


def filename_formatter(x):
    f, e = os.path.splitext(filename_formatter_re.sub('_', x))
    return "{}.{}".format(f[:70], e)


def filename_add_suffix(x, s):
    a, b = os.path.splitext(str(x))
    return "{}{}{}".format(a, s, b)


time_formatter = lambda: time.strftime("%Y%m%d-%H%M%S")
total_texts = xinqing_texts


class Toolbox:
    def __init__(self, datasets_root, enc_models_dir, syn_models_dir, voc_models_dir, toolbox_files_dir, low_mem):
        sys.excepthook = self.excepthook

        self._out_dir = Path(toolbox_files_dir)
        self.make_out_dirs()

        self.datasets_root = datasets_root
        self.datasets = [p.name for p in Path(datasets_root).glob("*") if p.is_dir()]

        metapath = Path(self.datasets_root).joinpath("metadata.csv")
        if metapath.is_file():
            itdt = {}
            for line in open(metapath, encoding="utf8"):
                parts = line.strip().split("\t")
                idx, text = parts[0], parts[1]
                itdt[idx] = text
            self.itdt = itdt
        else:
            self.itdt = {}

        self.low_mem = low_mem
        self.utterances = set()
        self.current_generated = (None, None, None, None)  # speaker_name, spec, breaks, wav

        self.synthesizer = None  # type: Synthesizer

        # Initialize the events and the interface
        self.ui = UI()
        self.reset_ui(enc_models_dir, syn_models_dir, voc_models_dir)
        self.setup_events()
        self.ui.start()

    def make_out_dirs(self):
        self._out_dir.mkdir(exist_ok=True)

        self._out_mel_dir = self._out_dir.joinpath('mels')
        self._out_mel_dir.mkdir(exist_ok=True)

        self._out_wav_dir = self._out_dir.joinpath('wavs')
        self._out_wav_dir.mkdir(exist_ok=True)

        self._out_embed_dir = self._out_dir.joinpath('embeds')
        self._out_embed_dir.mkdir(exist_ok=True)

        self._out_record_dir = self._out_dir.joinpath('records')
        self._out_record_dir.mkdir(exist_ok=True)

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)

    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     self.datasets,
                                                                     level)
        text_func = lambda: self.ui.text_prompt.setPlainText(np.random.choice(total_texts))
        self.ui.random_dataset_button.clicked.connect(text_func)
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))

        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)

        def func():
            self.synthesizer = None

        self.ui.synthesizer_box.currentIndexChanged.connect(func)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)

        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)
        self.ui.take_generated_button.clicked.connect(self.preprocess)

        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.compare_button.clicked.connect(self.compare)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir):
        self.ui.populate_browser(self.datasets_root, self.datasets, 0, True)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)

    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root, self.ui.current_dataset_name, self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            # name = '/'.join(fpath.relative_to(self.datasets_root).parts)
            dat = self.ui.current_dataset_name.replace("\\", "#").replace("/", "#")
            spk = self.ui.current_speaker_name.replace("\\", "#").replace("/", "#")
            aud = self.ui.current_utterance_name.replace("\\", "#").replace("/", "#")
            speaker_name = "#".join((dat, spk))
            name = "#".join((speaker_name, aud))
            # name = '-'.join(fpath.relative_to(self.datasets_root.joinpath(self.ui.current_dataset_name)).parts)
            # speaker_name = self.ui.current_speaker_name.replace("\\", "-").replace("/", "-")
            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)

    def compare(self):
        """
        1.判断参考音频是否有对应文本。
        2.输入框更新为参考文本。
        3.合成参考音频对应文本的语音。
        4.展示embed,spectrogram,alignment。
        :return:
        """
        idx = self.ui.selected_utterance.name.replace("#", "/")
        idx = re.sub(r"(_preprocessed)(\..*?)$", r"\2", idx)
        if idx not in self.itdt:
            print("Compare Failed! index: {}".format(idx))
            return

        self.ui.text_prompt.setPlainText(self.itdt[idx])
        self.synthesize()
        self.vocode()

    def preprocess(self):
        wav = self.ui.selected_utterance.wav
        out = remove_noise(wav, sr=Synthesizer.sample_rate)
        hp = Dict2Obj({})
        hp["vad_window_length"] = 10  # milliseconds
        hp["vad_moving_average_width"] = 2
        hp["vad_max_silence_length"] = 2
        hp["audio_norm_target_dBFS"] = -32
        hp["sample_rate"] = 16000
        hp["int16_max"] = (2 ** 15) - 1
        out = trim_long_silences(out, hparams=hp)

        spec = Synthesizer.make_spectrogram(out)
        self.ui.draw_align(spec[::-1], "current")

        # name = filename_add_suffix(self.ui.selected_utterance.name, "_preprocessed")
        # speaker_name = self.ui.selected_utterance.speaker_name
        # self.add_real_utterance(out, name, speaker_name)

    def record(self):
        wav = self.ui.record_one(encoder.sampling_rate, 5)
        if wav is None:
            return

        self.ui.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_{}".format(time_formatter())
        fpath = self._out_record_dir.joinpath(name + '.wav')
        save_wav(wav, fpath, encoder.sampling_rate)  # save
        wav = Synthesizer.load_preprocess_wav(fpath)  # 保持一致的数据格式

        self.add_real_utterance(wav, name, speaker_name)

    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        np.save(self._out_embed_dir.joinpath(name + '.npy'), embed, allow_pickle=False)  # save

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)

    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)

    def synthesize(self):
        self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)

        # Synthesize the spectrogram
        if self.synthesizer is None:
            model_dir = Path(self.ui.current_synthesizer_model_dir)
            checkpoints_dir = model_dir.joinpath("checkpoints")
            hp_path = model_dir.joinpath("metas", "hparams.json")  # load from trained models
            if hp_path.exists():
                hparams = Dict2Obj(json.load(open(hp_path, encoding="utf8")))
            else:
                hparams = None
            self.synthesizer = Synthesizer(checkpoints_dir, low_mem=self.low_mem, hparams=hparams)
        if not self.synthesizer.is_loaded():
            self.ui.log("Loading the synthesizer %s" % self.synthesizer.checkpoint_fpath)

        ptext = self.ui.text_prompt.toPlainText()
        texts = ptext.split("\n")

        embed = self.ui.selected_utterance.embed
        embeds = np.stack([embed] * len(texts))
        specs, aligns = self.synthesizer.synthesize_spectrograms(texts, embeds, return_alignments=True)

        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        align = np.concatenate(aligns, axis=1)

        fref = self.ui.selected_utterance.name
        ftext = '。'.join(texts)
        ftime = '{}'.format(time_formatter())
        fname = filename_formatter('{}_{}_{}zi_{}.npy'.format(fref, ftime, len(ftext), ftext))
        np.save(self._out_mel_dir.joinpath(fname), spec, allow_pickle=False)  # save

        self.ui.draw_spec(spec, "generated")
        self.ui.draw_align(align, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Synthesize the waveform
        if not vocoder.is_loaded():
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            self.ui.log(line, "overwrite")
            self.ui.set_loading(i, seq_len)

        wav = None
        vocname = ""
        if self.ui.current_vocoder_fpath is not None:
            model_fpath = self.ui.current_vocoder_fpath
            vocname = Path(model_fpath).parent.stem

            if Path(model_fpath).parent.stem == "melgan":
                self.ui.log("Waveform generation with MelGAN... ")
                wav = vocoder_melgan.infer_waveform_melgan(spec, model_fpath)
            elif Path(model_fpath).parent.stem == "wavernn":
                self.ui.log("Waveform generation with WaveRNN... ")
                wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)

        if wav is None:
            vocname = "griffinlim"
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)

        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        self.ui.play(wav, Synthesizer.sample_rate)

        fref = self.ui.selected_utterance.name
        ftime = '{}'.format(time_formatter())
        ftext = self.ui.text_prompt.toPlainText()
        fms = int(len(wav) * 1000 / Synthesizer.sample_rate)
        fvoc = vocname
        fname = filename_formatter('{}_{}_{}_{}ms_{}.wav'.format(fref, ftime, fvoc, fms, ftext))
        save_wav(wav, self._out_wav_dir.joinpath(fname), Synthesizer.sample_rate)  # save

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()

        encoder_wav = preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        name = speaker_name + "_gen_{}".format(time_formatter())
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)

        np.save(self._out_embed_dir.joinpath(name + '.npy'), embed, allow_pickle=False)  # save

        self.utterances.add(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "generated")
        self.ui.draw_umap_projections(self.utterances)

    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath

        self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return
        else:
            self.ui.log("Loading the vocoder %s... " % model_fpath)
            self.ui.set_loading(1)
            start = timer()

            if Path(model_fpath).parent.stem == "melgan":
                vocoder_melgan.load_vocoder_melgan(model_fpath)
            elif Path(model_fpath).parent.stem == "wavernn":
                vocoder.load_model(model_fpath)
            else:
                return

            self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
            self.ui.set_loading(0)
