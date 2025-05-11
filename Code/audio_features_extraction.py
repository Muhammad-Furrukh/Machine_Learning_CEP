# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:31:52 2025

@author: DELL
"""

import librosa
import numpy as np
import os

np.random.seed(42)

# Computes audio features from the fed sample
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Delta MFCCs
    delta_mfcc = librosa.feature.delta(mfccs)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    feature_vector = np.hstack([
        mfccs_mean,          # 13
        delta_mfcc_mean,     # 13
        chroma_mean,         # 12
        centroid_mean,       # 1
        bandwidth_mean,      # 1
        zcr_mean,            # 1
        rms_mean,            # 1
        duration             # 1
    ])

    return feature_vector

# To randomly arrange samples in the dataset 
def randomize (data, labels):
    n  = len(data)
    rand_order = np.random.permutation(n)
    shuffled_data = data[rand_order]
    shuffled_labels = labels[rand_order]
    
    return shuffled_data, shuffled_labels

audio_files = []
for i in range(1, 251):
    for char in ['c', 'd', 'g']:
        audio_files.append("../Samples/" + char + str(i) + ".wav")

features_list = []

for file in audio_files:
    features = extract_features(file)
    features_list.append(features)

labels = []
for n in range(750):
    labels.append(n%3)

features_list = np.array(features_list)
labels = np.array(labels)

train_data = features_list[:540]
train_labels = labels[:540]
test_data = features_list[540:]
test_labels = labels[540:]

# Random sorting done after splitting to ensure equal samples of each category
train_data, train_labels = randomize(train_data, train_labels)
test_data, test_labels = randomize(test_data, test_labels)
audio_features = {"train_data": train_data, "train_labels": train_labels, "test_data": test_data, "test_labels": test_labels}

save_path = os.path.join("../Dataset", 'audio_feature.npz')
np.savez(save_path, **audio_features)