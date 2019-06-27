#!/usr/bin/env python3

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


from PIL import Image
from xeger import Xeger

from morselib import *

#print(soundfile.available_formats())
#print(soundfile.available_subtypes())
#print(soundfile.default_subtype('OGG'))

def write_file(text, base,
        maxlength=30,
        samplerate=4410, 
        snr=20,
        wpm=25,
        tone=600,
        fading_strength=0.0,
        fading_cycle=1.5,
        fading_init=0,
        use_bandpass=False,
        n = 0,
        total = 0,
        ):

    data, label = generate_cw_data_with_snr(
            text,
            maxlength=maxlength,
            samplerate=samplerate,
            snr=snr,
            wpm=wpm,
            tone=tone,
            fading_strength=fading_strength,
            fading_cycle=fading_cycle,
            fading_init=fading_init,
            use_bandpass=use_bandpass,
            )

    # write labels as PNG
    label_file = pathlib.Path(base + '-keying.png')
    label_file.absolute().parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label).save(label_file)
    #print(base + '-keying.png written')

    soundfile.write(base + '.ogg', data * 0.5, samplerate, format='OGG')
    print('{:04d}/{:04d} {}.ogg written'.format(n, total, base))

snr = 20
wpm = 40
tone = 600
write_file(
        "CQ CQ DE JH1UMV JH1UMV PSE K",
        "/tmp/dataset/test-{}dB-{}wpm-{}Hz".format(snr, wpm, tone),
        maxlength=20,
        snr=snr,
        wpm=wpm,
        tone=tone,
        fading_strength=0.01,
        fading_cycle=1.5,
        use_bandpass=True,
        )

for snr in (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10, 20, 40):
    wpm = 25
    tone = 600
    write_file(
            "CQ CQ DE JH1UMV JH1UMV PSE K",
            "/tmp/dataset/cw-samples/{}dB-{}wpm-{}Hz".format(snr, wpm, tone),
            maxlength=20,
            snr=snr,
            wpm=wpm,
            tone=tone,
            use_bandpass=True,
            )

string_random = Xeger(limit=10).xeger
executor = concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

phase_count = 2
snr = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40)
wpm = (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40)
config = list(itertools.product(wpm, snr))

total = phase_count * len(config)
n = 0
for phase in range(phase_count):
    for wpm, snr in config:
        n += 1
        init_random_seed(snr << 24 | wpm << 16 | phase)
        text = " ".join(  string_random('[KMURESNAPTLWI.JZ=FOY,VG5/Q92H38B?47C1D60X]+') for n in range(20) )
        # print(wpm, snr, text)
        tone = 600
        executor.submit(
                write_file, 
                text,
                "/tmp/dataset/cw-train/{:03d}-{}dB-{}wpm-{}Hz".format(phase, snr, wpm, tone),
                maxlength=30,
                snr=snr,
                wpm=wpm,
                tone=tone,
                n=n,
                total=total,
                )
