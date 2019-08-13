#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dependent Libraries

import os
import numpy as np
import pandas as pd
import re
from scipy.io import wavfile

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter

import statsmodels.api as smt
import seaborn as sns
from scipy.signal import spectrogram
import heapq


def run_train(ts, length, num_peaks, random = True, num_random_crops = 100, smoothing_window = 101):
    # ts is a one dimensional time series array
    # window is the size of the crop window
    # num_peaks is the number of peaks to calculate
    if random:
        crops = rand_crops(ts, length, num_random_crops)
    else:
        k = len(ts) / length
        crops = seq_crops(ts, k)
    peaks, means, stds = build_dist(crops, num_peaks, smoothing_window)
    params = [means, stds, length, num_peaks]
    return(params)



def run_predict(ts, params, threshold = 0.5):
    # ts is a one dimensional time series array
    # params is list output from run_train()
    # threshold is a dummy placeholder for when the true generating frequency(ies) is unknown
    cavitation = False
    ts = ts[0:params[2]]
    seq = seq_crops(ts, 1)
    peaks, means, stds = build_dist(seq, params[3])
    out = detect_outlier(ts, params[0], params[1], params[2], threshold)
    return(out)




def rand_crops(ts, window, k):
    crops = {}
    for crop in range(k):
        start = np.random.randint(0, len(ts))
        crops[str(crop)] = [0, [ts[start:(start + window)]]]
    return(crops)

def seq_crops(ts, k):
    crops = {}
    window = int(len(ts) / k)
    start = 0
    for crop in range(k):
        crops[str(crop)] = [0, [ts[start:(start + window)]]]
        start += window
    return(crops)

def build_dist(dic, num_peaks = 3, window = 101, polyorder = 5, ratio = 1/32, smooth = True):
    # Assume everything is normal
    peaks = []
    for crop in dic.keys():
        ts = dic[crop][1][0]
        N = int(len(ts) * ratio)
        nfft = np.abs(np.fft.fft(ts))[0:N]
        if smooth == True and len(nfft) > window:
            savgol_filter(nfft, window, polyorder)
        peaks.append(np.argpartition(nfft, -num_peaks)[-num_peaks:])
    peaks = np.array(peaks)
    peaks.sort(axis = 1)
    means = np.mean(peaks, axis = 0)
    sds = np.std(peaks, axis = 0)
    return(peaks, means, sds)

def detect_outlier(ts, dist_means, dist_stds, window, alpha = 0.003, threshold = 0.5):
    k = int(len(ts) / window)
    subsets = seq_crops(ts, k)
    samples, means, stds = build_dist(subsets, num_peaks = dist_means.shape[0])
    outputs = []
    for col in range(dist_means.shape[0]):
        high_bound = dist_means[col] + (3 * dist_stds[col])
        low_bound = dist_means[col] - (3 * dist_stds[col])
        column = samples[:, col]
        out = (column > high_bound) + (column < low_bound)
        outputs.append(out)
    outputs = np.sum(np.array(outputs), axis = 0)
    return(outputs >= np.ceil(threshold * dist_means.shape[0]))

def readwav(filename):
    fs, data = wavfile.read("data/" + filename)
    data = pd.DataFrame(data)
    return(fs, data)

def createdict(foldername):
    dic = {}
    files = os.listdir("data/" + foldername)
    for file in files:
        dic[file] = readwav(foldername + '/' + file)
    return dic, files

def plotsounds(dic):
    rows = len(dic.keys())
    
    grid = (rows, 1)
    
    for i, file in enumerate(dic.keys()):
        ax = plt.subplot2grid(grid, (i, 0))
        dic[file][1][0].plot(ax = ax, figsize = (20, 25), c = tuple(np.random.rand(1, 3).flatten().tolist()))
        ax.set_title(file, fontsize = 16, fontweight = 'bold')
        sns.despine()
        plt.show()
    
    return(ax)

def plotspecs(dic):
    rows = len(dic.keys())
    
    grid = (rows, 1)
    
    for i, file in enumerate(dic.keys()):
        fs = dic[file][0]
        ts = dic[file][1][0]
        plt.figure(figsize = (20, 32))
        ax = plt.subplot2grid(grid, (i, 0))
        ax.set_title(file, fontsize = 16, fontweight = 'bold')
        plt.specgram(ts, Fs = fs)
        sns.despine()
    
    return(ax)

def plotfreq(dic, cutoff = 50000, half = False, smooth = False, window = 501, polyorder = 5):
    rows = len(dic.keys())
    
    grid = (rows, 1)
    
    for i, file in enumerate(dic.keys()):
        ax = plt.subplot2grid(grid, (i, 0))
        ts = dic[file][1][0]
        if half == True:
            cutoff = int(len(ts) / 2)
        fft = np.abs(np.fft.fft(ts))[0:cutoff]
        if smooth == True:
            savgol_filter(fft, window, polyorder)
        pd.DataFrame(fft).plot(ax = ax, figsize = (20, 25), c = tuple(np.random.rand(1, 3).flatten().tolist()))
        ax.set_title(file, fontsize = 16, fontweight = 'bold')
        sns.despine()
        plt.show()
    
    return(ax)






