import pandas as pd
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def load_file(filepath):
    dataframe = pd.read_csv(filepath, delim_whitespace=True)
    part_of_experiment = dataframe['annotation'] != 0
    cleaned_dataframe = dataframe[part_of_experiment]
    # cleaned_dataframe.index = pd.to_datetime(cleaned_dataframe['time'], unit='ms')
    return cleaned_dataframe


def windowed_view(dataframe, interval, overlap):
    windows = []
    start_time = 0
    end_time = 700000
    while (start_time < end_time):
        window_end = start_time + interval
        windows.append(dataframe[(dataframe['time'] > start_time) & (dataframe['time'] < window_end)])
        start_time += interval - overlap

    return windows


def remove_pre_FOG(window, period):
    fog_begin = True
    for index, row in window.iterrows():
        if row['annotation'] == 1 & fog_begin:
            window = window.iloc[:index - period, index:]
            fog_begin = False
        else:
            fog_begin = True
    return window

def extract_supervised_features(window, sampling_rate):
    # Creates a vector of features assosciated with the input window signal
    col_names = ['freezingEnergySum', 'LocomotoryEnergySum', 'FreezingIndex']
    supervised_features = pd.DataFrame(columns=col_names)
    print(window['time'])
    xf = np.linspace(0.0, 10 , 63)
    yf = np.fft.fft(window['shank_x'])

    plt.plot(xf, yf)
    plt.show()
    return supervised_features

