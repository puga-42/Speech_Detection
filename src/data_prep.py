'''
Defines a class to:
1. take in audio and text data
2. extract necessary features

'''

import glob
import eng_to_ipa as ipa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import librosa
import librosa.display
import soundfile as sf
import IPython.display as ipd

from src.phonemes import char_map, index_map


# class AudioGenerator():
#     def __init__(self, step=10, window=20, max_freq=8000,
#         minibatch_size=20, desc_file=None, spectrogram=True, max_duration=10.0, 
#         sort_by_duration=False):
#         """
#         Params:
#             step (int): Step size in milliseconds between windows (for spectrogram ONLY)
#             window (int): FFT window size in milliseconds (for spectrogram ONLY)
#             max_freq (int): Only FFT bins corresponding to frequencies between
#                 [0, max_freq] are returned (for spectrogram ONLY)
#             desc_file (str, optional): Path to a JSON-line file that contains
#                 labels and paths to the audio files. If this is None, then
#                 load metadata right away
#         """


def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


def get_audio_and_text_data(base_path):
    text_files = glob.glob(base_path + "/**/*.txt", recursive = True)  # get list of all text files in dir
    audio_paths, durations, texts = [], [], []
    ## Append to audio_paths, texts;
    for i in range(len(text_files)):
        ## read text line by line

        with open(text_files[i]) as f:
            content = f.readlines()
            idx =  [char[:16] for char in content]            
            idx = [base_path + index[:4] + '/' + index[5:11] + '/' + index + '.flac' for index in idx]
            # text = [letter[17:-1] for letter in content]
            # text = [sentence.lower() for sentence in text]
            # text = [ipa.convert(sentence) for sentence in text]

            #append paths and texts
            audio_paths += idx
            # texts += text


        if i%50 == 0:
            print('text file: ', i)

    return audio_paths, texts
    

from numpy.lib.stride_tricks import as_strided

def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram
    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).
    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x
    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs

def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    
    data, sample_rate = librosa.load(filename)

    if data.ndim >= 2:
        data = np.mean(data, 1)
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of "
                         " sample rate")
    if step > window:
        raise ValueError("step size must not be greater than window size")
    hop_length = int(0.001 * step * sample_rate)
    fft_length = int(0.001 * window * sample_rate)
    pxx, freqs = spectrogram(
        data, fft_length=fft_length, sample_rate=sample_rate,
        hop_length=hop_length)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))



def sort_by_duration(durations, audio_paths, texts):
    return zip(*sorted(zip(durations, audio_paths, texts)))



def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = index_map[c]
        text.append(ch)
    return text


