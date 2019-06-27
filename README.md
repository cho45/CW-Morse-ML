# CW-Morse-ML
Denoising / Recognizing CW morse code with machine learning

# Example

SNR=2dB (noise bandwidth = 500Hz) https://soundcloud.com/cho45/test-2db
![denoising sample](./docs/denoise.png)

SNR=10dB (noise bandwidth = 500Hz)
![denoising sample](./docs/denoise-10dB.png)

# Resources

 * [LoadData-Sample.ipynb](./LoadData-Sample.ipynb) Example of loading training wav with labels (Jupyter notebook)
 * [morselib.py](./morselib.py) some utility functions
 * [denoise/train.py](./denoise/train.py) example denoising training script (keras)
