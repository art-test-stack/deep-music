import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa


def show_signal_wave(data_path):

    x_1, fs = librosa.load(data_path)

    ig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    librosa.display.waveshow(x_1, sr=fs, ax=ax)
    ax.set(title='Signal wave')
    ax.label_outer()

    plt.show()


def extract_chroma_features(data_path):

    x_1, fs = librosa.load(data_path)
    hop_length = 1024

    x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs,
                                            hop_length=hop_length)

    fig, ax = plt.subplots(nrows=1, sharey=True)
    img = librosa.display.specshow(x_1_chroma, x_axis='time',
                                y_axis='chroma',
                                hop_length=hop_length, ax=ax)
    ax.set(title='Chroma Representation of Wave')
    fig.colorbar(img, ax=ax)

    plt.show()
