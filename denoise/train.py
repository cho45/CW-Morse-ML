#!/usr/bin/env python3
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import glob
import random
import json
import pathlib
import datetime
import gc
import sys


from numpy import array
from PIL import Image # pip install Pillow
import librosa

import keras
from keras.utils import plot_model
from keras.layers import Input, Dense, GRU, Dropout, RepeatVector, BatchNormalization, Activation, LSTM, Bidirectional, Masking, CuDNNGRU, TimeDistributed, MaxPooling1D, Conv1D, GlobalAveragePooling1D, Flatten, UpSampling1D, LocallyConnected1D, Reshape
from keras.layers import LeakyReLU, Add
from keras.models import Model
import keras_metrics as km

sys.path.append("../")

from morselib import *

keras.backend.clear_session()

SAMPLE_RATE = 210
MAX_DURATION_SEC = 30
MAX_SAMPLES = 6000

timestep = 630
batch_size = 64
features = 2

stateful = False
inputs = Input(batch_shape=(None, timestep, features))
x = inputs

trainable = True
x = Conv1D(64, 2, padding='same', activation=None, name='input64a', trainable=trainable)(x)
x = BatchNormalization(name='bn_in', trainable=trainable)(x)
x = LeakyReLU()(x)
count = 5
for n in range(count):
    s = x
    x = BatchNormalization(name='bn_in2a/{}'.format(n), trainable=trainable)(x)
    x = LeakyReLU()(x)
    x = Conv1D(16, 1, padding='same', activation=None, name='conv1d_0/16/16/{}'.format(n), trainable=trainable)(x)
    x = BatchNormalization(name='bn_in1a/{}'.format(n), trainable=trainable)(x)
    x = LeakyReLU()(x)
    x = Conv1D(16, 3, padding='same', activation=None, name='conv1d_1/16/16/{}'.format(n), trainable=trainable)(x)
    x = BatchNormalization(name='bn_in0a/{}'.format(n), trainable=trainable)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)
    x = Conv1D(64, 1, padding='same', activation=None, name='conv1d_2a/64/64/{}'.format(n), trainable=trainable)(x)
    x = Add()([x, s])

x = TimeDistributed(Dense(1, activation='sigmoid'), name='tmout', trainable=trainable)(x)

outputs = x
model = Model(inputs, outputs)
# Adam lr = 1.0 by default
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', km.categorical_precision(), km.categorical_recall()])
#model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.2, decay=0.0, nesterov=True), loss='binary_crossentropy', metrics=['accuracy', km.categorical_precision(), km.categorical_recall()])

if True:
    models = glob.glob("./model/*.hdf5")
    models.sort(key=os.path.getmtime)
    print('load', models[-1])
    model.load_weights(models[-1], by_name=True)
    # model.load_weights(models[-1])

plot_model(model, to_file="model.png", show_shapes=True)


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

# https://keras.io/ja/utils/#sequence
class DataSet(keras.utils.Sequence):
    def __init__(self, files, split=1, stride=1):
        files = files[: math.floor(len(files) / split)*split ]

        self.epochs = 0
        self.splitted = [files[i::split] for i in range(split)]
        self.files = self.splitted[0]
        self.batch_per_file = math.ceil(MAX_SAMPLES / batch_size / stride)
        self.loaded = {}
        self.stride = stride

    def __getitem__(self, index):
        """
        return one batch
        """

        file_index = math.floor(index / self.batch_per_file)
        batch_index = index % self.batch_per_file

        file = self.files[file_index]
        if file not in self.loaded:
            # print('  {} load file {} {}     '.format(os.getpid(), file, batch_index))
            self.loaded[file] = {}
            self.loaded[file]['data'] = MorseData(MorseWavData.load(file), batch_size, self.stride)

        loaded = self.loaded[file]
        data = loaded['data']

        if len(data) == 0:
            print(file)
            raise Exception(file)

        batch_index = batch_index % len(data)
        x = data.get_x(batch_index)

        y = data.get_key(batch_index)

        return x, y


    def __len__(self):
        """
        return total batch size
        """
        return len(self.files) * self.batch_per_file

    def on_epoch_end(self):
        self.loaded = {}
        self.epochs += 1
        self.files = self.splitted[self.epochs % len(self.splitted)]
        gc.collect()

class RandomDataSet(keras.utils.Sequence):
    def __init__(self, count, snr, wpm, fading_strength=None):
        self.snr = snr
        self.wpm = wpm
        self.count = count
        self.fading_strength = None

    def __getitem__(self, index):
        snr = self.snr or random.uniform(0, 20)
        wpm = self.wpm or random.uniform(10, 40)
        fading_strength = self.fading_strength or random.uniform(0.0, 0.2)
        fading_cycle  = random.uniform(0.2, 1.0)
        fading_init = random.uniform(0, 10)
        data = MorseData(
                MorseWavData.generate(
                    snr=snr,
                    wpm=wpm,
                    fading_strength=fading_strength,
                    fading_cycle=fading_cycle,
                    fading_init=fading_init
                    ),
                MAX_SAMPLES, 
                151 
                )

        x = data.get_x(0)
        y = data.get_key(0)
        return x, y

    def __len__(self):
        return self.count


callbacks = [
    keras.callbacks.TensorBoard(log_dir="/tmp/tensorboard-log/{}-{}".format(pathlib.Path(__file__).resolve().parent.name, datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))),
    keras.callbacks.ModelCheckpoint(filepath = os.path.join("./model",'cnn_model{epoch:02d}-loss{loss:.2f}-val_acc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0001),
]

#test_files = glob.glob('/tmp/dataset/cw-test/n=*-keying.png')
#train_files = glob.glob('/tmp/dataset/cw-train/n=*-keying.png')

#test_gen = DataSet(random.sample(test_files, len(test_files)), 1, 97)
#train_gen = DataSet(random.sample(train_files, len(train_files)), 7, 97)

test_gen = RandomDataSet(100, None, None, fading_strength=0.0)
train_gen = RandomDataSet(1000, None, None)

# keras.backend.clear_session()
model.summary()
model.fit_generator(
    generator=train_gen,
    epochs=100,
    steps_per_epoch=len(train_gen),
    verbose=1,
    validation_data=test_gen,
    validation_steps=len(test_gen),
    callbacks=callbacks,
    workers=4,
    use_multiprocessing=True,
    shuffle=True,
    )
