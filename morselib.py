#!/usr/bin/env python3

import os
import json
import pathlib
import numpy as np
import librosa
import soundfile # pip install pysoundfile
import math
import scipy
import random
import keras
import re
import dataclasses

from xeger import Xeger
from PIL import Image


DEFINITION = json.loads(pathlib.Path(__file__).absolute().parent.joinpath("DEFINITION.json").read_text())
DEFINED_MORSE_CODES = tuple(DEFINITION["DEFINED_MORSE_CODES"])
EN_TO_MORSE =  {v: k for k, v in DEFINITION["EN"].items()} 

OUTPUT_FEATURES = len(DEFINED_MORSE_CODES)

DEFINED_MORSE_CODES_1HOT = keras.utils.to_categorical(np.arange(len(DEFINED_MORSE_CODES)), OUTPUT_FEATURES)  
morse_code_categorical = np.vectorize(lambda n: DEFINED_MORSE_CODES_1HOT[n], signature='()->(n)')


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def init_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

def create_morse_buffer(code, samplerate, wpm=20, tone=600, word_spacing=1.0, char_spacing=1.0):
    speed = 1200.0 / wpm
    unit = samplerate * (speed / 1000)
    tone = samplerate / (2 * np.pi * tone)

    dot_unit  = round(1 * unit)
    dash_unit = round(3 * unit)
    word_unit = round(7 * unit * word_spacing)
    char_unit = round(3 * unit * char_spacing)

    sequence = []
    length = 0.0
    for c in code:
        c = c.upper()
        if c == ' ':
            length += word_unit
            sequence.append({ 'c': ' ', 'n': -word_unit, 'intermediate': False })
        else:
            m = EN_TO_MORSE[c]
            for j, mc in enumerate(m):
                if mc == '.':
                    length += dot_unit
                    sequence.append({ 'c': m, 'n': dot_unit, 'intermediate': True })
                elif mc == '-':
                    length += dash_unit
                    sequence.append({ 'c': m, 'n': dash_unit, 'intermediate': True })

                if j < (len(m) - 1):
                    length += dot_unit
                    sequence.append({ 'c': m, 'n': -dot_unit, 'intermediate': True })

            length += char_unit
            sequence.append({ 'c': m, 'n': -char_unit, 'intermediate': False })

    length = math.ceil(length + char_unit + 1)

    data = np.zeros(length, dtype=np.float32)
    keying = np.zeros(length, dtype=np.uint8)
    codeid = np.zeros(length, dtype=np.uint8)
    recog  = np.zeros(length, dtype=np.uint8)

    x = 0
    for seq in sequence:
        c = DEFINED_MORSE_CODES.index(seq['c'])
        s = seq['n']
        if s < 0:
            s = -s
            data[x:x+s] = 0
            codeid[x:x+s] = c
            x += s

            if c != 0 and (not seq['intermediate']):
                s = char_unit
                recog[x:x+s] = c
        else:
            data[x:x+s] = np.sin(np.arange(s, dtype=np.float32) / tone)
            codeid[x:x+s] = c
            keying[x:x+s] = 255
            x += s

            # e = samplerate * 0.002
            # for f in range(round(e)):
            #    data[x - f] = data[x - f] * (f / e)

    return data, keying, codeid, recog

def generate_cw_data_with_snr(
        text,
        maxlength=30,
        samplerate=4410, 
        snr=20,
        wpm=25,
        tone=600,
        fading_strength=0.0,
        fading_cycle=1.5,
        fading_init=0,
        use_bandpass=False,
        ):

    #samplerate = 44100
    maxsample  = samplerate * maxlength
    #print('maxsample', maxsample)

    snr_offset500 = 10 * math.log10(samplerate / 2.0 / 500.0)

    snr_linear = math.pow(10, (snr - snr_offset500) / 20)
    noise_level   = 1.0 / (snr_linear + 1.0)
    gain = snr_linear * noise_level
    # print(samplerate, 'noise_level', noise_level, 'gain', gain, 'ratio', gain / noise_level, 'dB', math.log10(gain / noise_level) * 20)

    # std random floor noise
    data = np.zeros(maxsample, dtype=np.float32)
    data += np.random.normal(0, 1.0, maxsample) * noise_level

    start = samplerate * 1

    # create morse code data
    mdata, keying, codeid, recog = ( d[:maxsample-start] for d in create_morse_buffer(text, samplerate=samplerate, wpm=wpm, tone=tone) )

    # apply fading to cw
    if fading_strength != 0.0:
        fading_cycle_t = samplerate / (2 * np.pi * fading_cycle)
        fading = np.arange(len(mdata), dtype=np.float32)
        fading += fading_init
        fading = np.cos(fading / fading_cycle_t) * fading_strength + (1 - fading_strength)
        mdata *= fading

    # mix cw to buffer
    data[start:start+len(mdata)] += mdata * gain

    # create label data
    label = np.zeros( (3, maxsample), dtype=np.uint8)
    label[0,start:start+len(mdata)] = keying
    label[1,start:start+len(mdata)] = codeid
    label[2,start:start+len(mdata)] = recog

    # apply bandpass filter around tone frequency
    if use_bandpass:
        data = butter_bandpass_filter(data, tone-250, tone+250, samplerate, order=5)

    return data, label


class MorseWavData:
    @classmethod
    def load(cls, path):
        base = re.sub(r'(-keying\.png|\.ogg)$', '', path)

        wav_file = base + '.ogg' 
        png_file = base + '-keying.png'

        tone, = re.search(r'([0-9]+)Hz', base).groups()
        tone = int(tone)

        wav, sr = librosa.load(wav_file, sr=None, mono=False)
        y = np.array(Image.open(png_file))
        key, mid, rht = y[0], y[1], y[2]
        return cls(wav, sr, key, mid, rht, tone)

    @classmethod
    def generate(cls, maxlength=30, samplerate=4410, snr=None, wpm=None, fading_strength=0.0, fading_cycle=1.5, fading_init=0):
        if not snr:
            snr = random.uniform(0, 20)

        if not wpm:
            wpm = random.uniform(20, 40)

        tone = random.uniform(400, 1000)

        string_random = Xeger(limit=10).xeger
        text = " ".join(  string_random('[KMURESNAPTLWI.JZ=FOY,VG5/Q92H38B?47C1D60X]+') for n in range(20) )
        wav, y = generate_cw_data_with_snr(
                text,
                maxlength=maxlength,
                samplerate=samplerate, 
                snr=snr,
                wpm=wpm,
                tone=tone,
                fading_strength=fading_strength,
                fading_cycle=fading_cycle,
                fading_init=fading_init,
                use_bandpass=False,
                )

        key, mid, rht = y[0], y[1], y[2]
        return cls(wav, samplerate, key, mid, rht, tone)

    def __init__(self, wav, sr, key, mid, rht, tone):
        self.wav = wav
        self.samplerate = sr
        self.key = key
        self.mid = mid
        self.rht = rht
        self.tone = tone

    def get_iq(self, bandpass_width=50, rsr=210):
        tx = self.samplerate / (2 * np.pi * self.tone)
        down_sampling_factor = math.floor(self.samplerate / rsr)

        wav = self.wav
        if bandpass_width != 0:
            wav = butter_bandpass_filter(wav, self.tone-bandpass_width, self.tone+bandpass_width, self.samplerate, order=3)

        trange =  np.arange(len(wav)) / tx
        ch0 =  scipy.signal.decimate( wav * np.cos(trange) , down_sampling_factor, ftype="fir")
        ch1 =  scipy.signal.decimate( wav * np.sin(trange) , down_sampling_factor, ftype="fir")

        key, mid, rht = ( d[::down_sampling_factor] for d in (self.key, self.mid, self.rht) )
        l = min(len(ch0), len(key))
        return (d[:l] for d in (ch0, ch1, key, mid, rht) )


