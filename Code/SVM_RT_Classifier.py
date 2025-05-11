# -*- coding: utf-8 -*-
"""
Created on Sat May 10 03:37:48 2025

@author: hp
"""

import librosa
import numpy as np
import joblib
import sounddevice as sd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def extract_features(y, sr):
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
# Parameters
fs = 22500       # Sample rate
duration = 3   # seconds of audio
keep_features = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 18, 35, 39, 40]
predictions = []
true_labels = []
while (True): 
    label = input()
    if label == 'q':
        break
    else:
        true_labels.append(int(label))
        
    print("\nRecording... Speak now")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    plt.plot(audio)
    plt.title("Waveform of Recorded Audio")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
    
    audio = audio.flatten()
    
    features = extract_features(audio, fs)
    features = features[keep_features]
    
    # Scale and reshape for prediction
    feature_scaled = scaler.transform([features])
    prediction = svm_model.predict(feature_scaled)
    predictions.append(prediction[0])
    
    if prediction[0] == 0:
        print("Predicted class: Cat")
    elif prediction[0] == 1:
        print("Predicted class: Dog")
    else:
        print("Predicted class: Goat")
        
acc = accuracy_score(true_labels, predictions)
print(acc)