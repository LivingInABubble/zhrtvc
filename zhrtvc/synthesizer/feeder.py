import os
import threading
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from synthesizer.infolog import log
from synthesizer.utils.text import text_to_sequence

_batches_per_group = 8

from encoder import inference as encoder


def embed_utterance(wav, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath, device="cpu")
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    return embed


class Feeder:
    """
        Feeds batches of data into queue on a background thread.
    """

    def __init__(self, coordinator, metadata_filename, hparams):
        super(Feeder, self).__init__()
        self.encoder_path = Path(hparams.encoder_path)
        self._coord = coordinator
        self._hparams = hparams
        self._cleaner_names = hparams.cleaners
        self._train_offset = 0
        self._test_offset = 0

        # Load metadata
        self._audio_dir = os.path.join(os.path.dirname(metadata_filename), "audio")
        self._mel_dir = os.path.join(os.path.dirname(metadata_filename), "mels")
        self._embed_dir = os.path.join(os.path.dirname(metadata_filename), "embeds")
        with open(metadata_filename, encoding="utf8") as fin:
            self._metadata = []
            for line in tqdm(fin, ncols=50, mininterval=2):
                # 支持相对路径和绝对路径
                # ../data/samples/aliaudio/Aibao/005397.mp3|mel-aliaudio-Aibao-005397.mp3.npy|embed-aliaudio-Aibao-005397.mp3.npy|64403|254|他走近钢琴并开始演奏“祖国从哪里开始”。
                audio_path, mel_path, embed_path, audio_size, mel_size, text = line.strip().split("|")
                if not os.path.exists(audio_path):
                    audio_path = os.path.join(self._audio_dir, audio_path)
                if not os.path.exists(mel_path):
                    mel_path = os.path.join(self._mel_dir, mel_path)
                if not os.path.exists(embed_path):
                    embed_path = os.path.join(self._embed_dir, embed_path)

                if os.path.exists(audio_path) and os.path.exists(mel_path) and os.path.exists(embed_path):
                    self._metadata.append([audio_path, mel_path, embed_path, audio_size, mel_size, text])
                else:
                    print("Load data failed!")
                    print("data:", line)

            frame_shift_ms = hparams.hop_size / hparams.sample_rate
            hours = sum([int(x[4]) for x in self._metadata]) * frame_shift_ms / (3600)
            log("Loaded metadata for {} examples ({:.2f} hours)".format(len(self._metadata), hours))

        # Train test split
        if hparams.tacotron_test_size is None:
            assert hparams.tacotron_test_batches is not None

        test_size = (hparams.tacotron_test_size if hparams.tacotron_test_size is not None
                     else hparams.tacotron_test_batches * hparams.tacotron_batch_size)
        indices = np.arange(len(self._metadata))
        train_indices, test_indices = train_test_split(indices,
                                                       test_size=test_size,
                                                       random_state=hparams.tacotron_data_random_state)

        # Make sure test_indices is a multiple of batch_size else round up
        len_test_indices = self._round_down(len(test_indices), hparams.tacotron_batch_size)
        extra_test = test_indices[len_test_indices:]
        test_indices = test_indices[:len_test_indices]
        train_indices = np.concatenate([train_indices, extra_test])

        self._train_meta = list(np.array(self._metadata)[train_indices])
        self._test_meta = list(np.array(self._metadata)[test_indices])
        np.random.shuffle(self._train_meta)
        np.random.shuffle(self._test_meta)
        self.test_steps = len(self._test_meta) // hparams.tacotron_batch_size

        if hparams.tacotron_test_size is None:
            assert hparams.tacotron_test_batches == self.test_steps

        # pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        # explicitely setting the padding to a value that doesn"t originally exist in the spectogram
        # to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.
        # Mark finished sequences with 1s
        self._token_pad = 1.

        with tf.device("/cpu:0"):
            # Create placeholders for inputs and targets. Don"t specify batch size because we want
            # to be able to feed different batch sizes at eval time.
            self._placeholders = [
                tf.placeholder(tf.int32, shape=(None, None), name="inputs"),
                tf.placeholder(tf.int32, shape=(None,), name="input_lengths"),
                tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels),
                               name="mel_targets"),
                tf.placeholder(tf.float32, shape=(None, None), name="token_targets"),
                tf.placeholder(tf.int32, shape=(None,), name="targets_lengths"),
                tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None),
                               name="split_infos"),

                # SV2TTS
                tf.placeholder(tf.float32, shape=(None, hparams.speaker_embedding_size),
                               name="speaker_embeddings")
            ]

            # Create queue for buffering data
            queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32,
                                     tf.int32, tf.int32, tf.float32], name="input_queue")
            self._enqueue_op = queue.enqueue(self._placeholders)
            self.inputs, self.input_lengths, self.mel_targets, self.token_targets, \
            self.targets_lengths, self.split_infos, self.speaker_embeddings = queue.dequeue()

            self.inputs.set_shape(self._placeholders[0].shape)
            self.input_lengths.set_shape(self._placeholders[1].shape)
            self.mel_targets.set_shape(self._placeholders[2].shape)
            self.token_targets.set_shape(self._placeholders[3].shape)
            self.targets_lengths.set_shape(self._placeholders[4].shape)
            self.split_infos.set_shape(self._placeholders[5].shape)
            self.speaker_embeddings.set_shape(self._placeholders[6].shape)

            # Create eval queue for buffering eval data
            eval_queue = tf.FIFOQueue(1, [tf.int32, tf.int32, tf.float32, tf.float32,
                                          tf.int32, tf.int32, tf.float32], name="eval_queue")
            self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
            self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, \
            self.eval_token_targets, self.eval_targets_lengths, \
            self.eval_split_infos, self.eval_speaker_embeddings = eval_queue.dequeue()

            self.eval_inputs.set_shape(self._placeholders[0].shape)
            self.eval_input_lengths.set_shape(self._placeholders[1].shape)
            self.eval_mel_targets.set_shape(self._placeholders[2].shape)
            self.eval_token_targets.set_shape(self._placeholders[3].shape)
            self.eval_targets_lengths.set_shape(self._placeholders[4].shape)
            self.eval_split_infos.set_shape(self._placeholders[5].shape)
            self.eval_speaker_embeddings.set_shape(self._placeholders[6].shape)

    def start_threads(self, session):
        self._session = session
        thread = threading.Thread(name="background", target=self._enqueue_next_train_group)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

        thread = threading.Thread(name="background", target=self._enqueue_next_test_group)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

    def _get_test_groups(self):
        meta = self._test_meta[self._test_offset]
        self._test_offset += 1
        return self.get_example(meta)

    def make_test_batches(self):
        start = time.time()

        # Read a group of examples
        n = self._hparams.tacotron_batch_size
        r = self._hparams.outputs_per_step

        # Test on entire test set
        examples = [self._get_test_groups() for i in tqdm(range(len(self._test_meta)), ncols=50, mininterval=2)]

        # Bucket examples based on similar output sequence length for efficiency
        examples.sort(key=lambda x: x[-1])
        batches = [examples[i: i + n] for i in range(0, len(examples), n)]
        np.random.shuffle(batches)

        log("\nGenerated %d test batches of size %d in %.3f sec" % (len(batches), n, time.time() - start))
        return batches, r

    def _enqueue_next_train_group(self):
        while not self._coord.should_stop():
            start = time.time()

            # Read a group of examples
            n = self._hparams.tacotron_batch_size
            r = self._hparams.outputs_per_step
            examples = [self._get_next_example() for i in tqdm(range(n * _batches_per_group), ncols=50, mininterval=2)]

            # Bucket examples based on similar output sequence length for efficiency
            examples.sort(key=lambda x: x[-1])
            batches = [examples[i: i + n] for i in range(0, len(examples), n)]
            np.random.shuffle(batches)

            log("\nGenerated {} train batches of size {} in {:.3f} sec".format(len(batches), n, time.time() - start))
            for batch in batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _enqueue_next_test_group(self):
        # Create test batches once and evaluate on them for all test steps
        test_batches, r = self.make_test_batches()
        while not self._coord.should_stop():
            for batch in test_batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
                self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        """Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
        """
        if self._train_offset >= len(self._train_meta):
            self._train_offset = 0
            np.random.shuffle(self._train_meta)

        meta = self._train_meta[self._train_offset]
        self._train_offset += 1
        return self.get_example(meta)

    def get_example(self, meta):
        text = meta[5]
        input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
        mel_target = np.load(meta[1])
        token_target = np.zeros(len(mel_target) - 1)  # np.asarray([0.] * (len(mel_target) - 1))
        embed_target = np.load(meta[2])
        return input_data, mel_target, token_target, embed_target, len(mel_target)

    def _prepare_batch(self, batches, outputs_per_step):
        assert 0 == len(batches) % self._hparams.tacotron_num_gpus
        size_per_device = int(len(batches) / self._hparams.tacotron_num_gpus)
        np.random.shuffle(batches)

        inputs = None
        mel_targets = None
        token_targets = None
        targets_lengths = None
        split_infos = []

        targets_lengths = np.asarray([x[-1] for x in batches], dtype=np.int32)  # Used to mask loss
        input_lengths = np.asarray([len(x[0]) for x in batches], dtype=np.int32)

        for i in range(self._hparams.tacotron_num_gpus):
            batch = batches[size_per_device * i:size_per_device * (i + 1)]
            input_cur_device, input_max_len = self._prepare_inputs([x[0] for x in batch])
            inputs = np.concatenate((inputs, input_cur_device), axis=1) if inputs is not None else input_cur_device
            mel_target_cur_device, mel_target_max_len = self._prepare_targets([x[1] for x in batch], outputs_per_step)
            mel_targets = np.concatenate((mel_targets, mel_target_cur_device),
                                         axis=1) if mel_targets is not None else mel_target_cur_device

            # Pad sequences with 1 to infer that the sequence is done
            token_target_cur_device, token_target_max_len = self._prepare_token_targets([x[2] for x in batch],
                                                                                        outputs_per_step)
            token_targets = np.concatenate((token_targets, token_target_cur_device),
                                           axis=1) if token_targets is not None else token_target_cur_device
            split_infos.append([input_max_len, mel_target_max_len, token_target_max_len])

        split_infos = np.asarray(split_infos, dtype=np.int32)

        ### SV2TTS ###

        embed_targets = np.asarray([x[3] for x in batches])

        ##############

        return inputs, input_lengths, mel_targets, token_targets, targets_lengths, \
               split_infos, embed_targets

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

    def _prepare_token_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets]) + 1
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_token_target(t, data_len) for t in targets]), data_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=self._pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode="constant", constant_values=self._target_pad)

    def _pad_token_target(self, t, length):
        return np.pad(t, (0, length - t.shape[0]), mode="constant", constant_values=self._token_pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder
