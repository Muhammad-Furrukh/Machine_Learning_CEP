# -*- coding: utf-8 -*-
"""
Created on Fri May  9 23:16:33 2025

@author: hp
"""

import librosa
import numpy as np
import joblib

def extract_features(file):
    y, sr = librosa.load(file, duration=3)
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

# Loading trained SVM model and fitted preprocessing scaler
svm_model = joblib.load('svm.pkl')
scaler = joblib.load('scaler.pkl')
keep_features = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 18, 35, 39, 40]

while (True):
    audio = input() # cX, dX or gX, where X is number from 1-250
    audio = "../Samples/" + audio + ".wav"

    features = extract_features(audio)
    features = features[keep_features]
    
    # Scale and reshape for prediction
    feature_scaled = scaler.transform([features])
    prediction = svm_model.predict(feature_scaled)
    
    if prediction[0] == 0:
        print("Predicted class: Cat")
    elif prediction[0] == 1:
        print("Predicted class: Dog")
    else:
        print("Predicted class: Goat")



