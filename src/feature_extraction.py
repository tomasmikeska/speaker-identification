import numpy as np
from scipy import signal
import python_speech_features
from sklearn import preprocessing


MFCC_COEFS = 24
SILENCE_WINDOW_SIZE = 0.0125 # seconds
OUTPUT_DIM = (0, 24) # 0 stands for variable length


def strip_init_err(s, freq):
    '''Strip initial recording error - 1.2s'''
    return s[int(freq*1.2):]

def normalize(x):
    return (x - x.mean()) / (x.max() - x.min())

def denoise(audio):
    return signal.medfilt(audio, 3) # Apply median filtering

def energy(samples):
    return np.sum(np.power(samples, 2)) / float(len(samples))

def remove_silence(audio, freq):
    output = np.array([])
    base_e = energy(audio)
    win_size = int(freq * SILENCE_WINDOW_SIZE)
    i = 0
    while (i + 1) * win_size < audio.shape[0]:
        window = audio[i*win_size:(i+1)*win_size]
        e = energy(window)
        if e > base_e * 0.2:
            output = np.append(output, window)
        i += 1
    return output

def mfcc(audio, freq):
    mfcc_data = python_speech_features.mfcc(audio, freq, 0.025, 0.01, MFCC_COEFS, appendEnergy = True)
    return preprocessing.scale(mfcc_data)

def extract_features(audio, freq):
    audio = strip_init_err(audio, freq)
    audio = denoise(audio)
    audio = normalize(audio)
    audio = remove_silence(audio, freq)
    return mfcc(audio, freq)
