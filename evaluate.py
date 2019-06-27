#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile # pip install pysoundfile
import json
import pathlib
import math
import scipy
import random
import os
import itertools
import concurrent.futures
import multiprocessing
import keras
import keras_metrics as km

from morselib import *

class MorseData:
    def __init__(self, wav_data, batch_size, stride):
        ch0, ch1, key, mid, rht = wav_data.get_iq()

        mag = min_max(np.sqrt(ch0 * ch0 + ch1 * ch1))
        pha = np.arctan2(ch0, ch1)
        var = np.absolute(np.convolve(np.cos(pha) + np.sin(pha)*1j, np.ones(5) / 5.0, mode='same'))
        var = min_max(var)
        x = np.concatenate(
                (
                    mag.reshape( (len(mag), 1) ),
                    var.reshape( (len(var), 1) ),
                ), 
                axis=1 
            )

        key = (key.astype('float32') / 255).reshape( (len(key), 1) )

        start_index = random.randint(0, stride)

        self.tx = keras.preprocessing.sequence.TimeseriesGenerator(
                x,
                x,
                timestep,
                sampling_rate=1,
                stride=stride,
                start_index=start_index,
                end_index=None,
                shuffle=False,
                reverse=False,
                batch_size=batch_size
            )

        self.tkey = keras.preprocessing.sequence.TimeseriesGenerator(
                key,
                x,
                timestep,
                sampling_rate=1,
                stride=stride,
                start_index=start_index,
                end_index=None,
                shuffle=False,
                reverse=False,
                batch_size=batch_size
            )

        self.tmid = keras.preprocessing.sequence.TimeseriesGenerator(
                mid,
                x,
                timestep,
                sampling_rate=1,
                stride=stride,
                start_index=start_index,
                end_index=None,
                shuffle=False,
                reverse=False,
                batch_size=batch_size
            )

        self.trht = keras.preprocessing.sequence.TimeseriesGenerator(
                rht,
                x,
                timestep,
                sampling_rate=1,
                stride=stride,
                start_index=start_index,
                end_index=None,
                shuffle=False,
                reverse=False,
                batch_size=batch_size
            )
    def __len__(self):
        return len(self.tx)

    def get_x(self, index):
        return self.tx[index][0]

    def get_key(self, index):
        return self.tkey[index][0]

    def get_mid(self, index):
        return morse_code_categorical(self.tmid[index][0])

    def get_rht(self, index):
        return morse_code_categorical(self.trht[index][0])

class RandomDataSet(keras.utils.Sequence):
    def __init__(self, batch_size, snr, wpm):
        self.snr = snr
        self.wpm = wpm
        self.batch_size = batch_size

    def __getitem__(self, index):
        data = MorseData( MorseWavData.generate(snr=self.snr, wpm=self.wpm), 3000, 151 )

        x = data.get_x(0)
        y = data.get_key(0)
        return x, y

    def __len__(self):
        return self.batch_size


model = keras.models.load_model("./denoise/model/cnn_model34-loss0.06-val_acc0.98.hdf5", compile=False)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(),  km.binary_recall()])  

timestep = model.input.shape[1].value
features = model.input.shape[2].value

plt.figure(figsize=(29,20))

for wpm in (10, 25, 40):
    scores = []
    snrs = (20, 15, 10, 7, 5, 3, 1,  -1)
    for snr in snrs:
        score = model.evaluate_generator(RandomDataSet(10, snr, wpm), verbose=0)
        # [('loss', 0.21604108944456413), ('acc', 0.9204921375506769), ('precision', 0.953636276632863), ('recall', 0.8895832487920386)]
        print(snr, wpm, list(zip(model.metrics_names, score)))
        scores.append(score)

    plt.plot(snrs, list(s[1] for s in scores), label='acc {}wpm'.format(wpm))

plt.xlabel('SNR dB')
plt.ylabel('score')
plt.legend()
plt.grid()
plt.show()

