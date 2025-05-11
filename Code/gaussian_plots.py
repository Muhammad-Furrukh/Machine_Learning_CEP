# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:08:11 2025

@author: hp
"""

import librosa
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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

cat_files = []
dog_files = []
goat_files = []
for i in range(1, 151):
    cat_files.append("../Samples/c" + str(i) + ".wav")
    dog_files.append("../Samples/d" + str(i) + ".wav")
    goat_files.append("../Samples/g" + str(i) + ".wav")
    
cat_features = []
dog_features = []
goat_features = []

for animal in [cat_files, dog_files, goat_files]:
    for file in animal:
        features = extract_features(file)
        if animal == cat_files:
            cat_features.append(features)
        elif animal == dog_files:
            dog_features.append(features)
        elif animal == goat_files:
            goat_features.append(features)
            
cat_features = np.array(cat_features)
dog_features = np.array(dog_features)
goat_features = np.array(goat_features)

# Plotting Gaussians for different features
# MFCC's (0-12)
for i in range(13):
    
    cats_mfcci = cat_features[:, i]  # all MFCC[i] means across samples
    dogs_mfcci = dog_features[:, i]  
    goats_mfcci = goat_features[:, i]  

    # Fit Gaussian
    mu_cats, std_cats = norm.fit(cats_mfcci)
    mu_dogs, std_dogs = norm.fit(dogs_mfcci)
    mu_goats, std_goats = norm.fit(goats_mfcci)

    # Plot
    x_cats = np.linspace(min(cats_mfcci), max(cats_mfcci), 150)
    p_cats = norm.pdf(x_cats, mu_cats, std_cats)
    x_dogs = np.linspace(min(dogs_mfcci), max(dogs_mfcci), 150)
    p_dogs = norm.pdf(x_dogs, mu_dogs, std_dogs)
    x_goats = np.linspace(min(goats_mfcci), max(goats_mfcci), 150)
    p_goats = norm.pdf(x_goats, mu_goats, std_goats)
    plt.figure()
    plt.plot(x_cats, p_cats, 'k', linewidth=2, label="cats")
    plt.plot(x_dogs, p_dogs, 'r', linewidth=2, label="dogs")
    plt.plot(x_goats, p_goats, 'b', linewidth=2, label="goats")
    plt.title(f'MFCC{i}')
    plt.xlabel('MFCC{i} Mean Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
# Delta MFCC's (0-12)
for i in range(13, 26):
    cats_dmfcci = cat_features[:, i]  # all MFCC[i] means across samples
    dogs_dmfcci = dog_features[:, i]  
    goats_dmfcci = goat_features[:, i]  

    # Fit Gaussian
    mu_cats, std_cats = norm.fit(cats_dmfcci)
    mu_dogs, std_dogs = norm.fit(dogs_dmfcci)
    mu_goats, std_goats = norm.fit(goats_dmfcci)

    # Plot
    x_cats = np.linspace(min(cats_dmfcci), max(cats_dmfcci), 150)
    p_cats = norm.pdf(x_cats, mu_cats, std_cats)
    x_dogs = np.linspace(min(dogs_dmfcci), max(dogs_dmfcci), 150)
    p_dogs = norm.pdf(x_dogs, mu_dogs, std_dogs)
    x_goats = np.linspace(min(goats_dmfcci), max(goats_dmfcci), 150)
    p_goats = norm.pdf(x_goats, mu_goats, std_goats)
    plt.figure()
    plt.plot(x_cats, p_cats, 'k', linewidth=2, label="cats")
    plt.plot(x_dogs, p_dogs, 'r', linewidth=2, label="dogs")
    plt.plot(x_goats, p_goats, 'b', linewidth=2, label="goats")
    plt.title(f'Delta MFCC{i-13}')
    plt.xlabel('Delta MFCC{i-13} Mean Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Chroma (0-11)
for i in range(26, 38):
    cats_chromai = cat_features[:, i]  # all Chroma[i] means across samples
    dogs_chromai = dog_features[:, i]
    goats_chromai = goat_features[:, i]

    # Fit Gaussian
    mu_cats, std_cats = norm.fit(cats_chromai)
    mu_dogs, std_dogs = norm.fit(dogs_chromai)
    mu_goats, std_goats = norm.fit(goats_chromai)

    # Plot
    x_cats = np.linspace(min(cats_chromai), max(cats_chromai), 150)
    p_cats = norm.pdf(x_cats, mu_cats, std_cats)
    x_dogs = np.linspace(min(dogs_chromai), max(dogs_chromai), 150)
    p_dogs = norm.pdf(x_dogs, mu_dogs, std_dogs)
    x_goats = np.linspace(min(goats_chromai), max(goats_chromai), 150)
    p_goats = norm.pdf(x_goats, mu_goats, std_goats)

    plt.figure()
    plt.plot(x_cats, p_cats, 'k', linewidth=2, label="cats")
    plt.plot(x_dogs, p_dogs, 'r', linewidth=2, label="dogs")
    plt.plot(x_goats, p_goats, 'b', linewidth=2, label="goats")
    plt.title(f'Chroma {i-26}')
    plt.xlabel(f'Chroma {i-26} Mean Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Spectral Centroid
cats_centroid = np.array(cat_features[:, 38])
dogs_centroid = np.array(dog_features[:, 38])
goats_centroid = np.array(goat_features[:, 38])
     

# Fit Gaussian
mu_cats, std_cats = norm.fit(cats_centroid)
mu_dogs, std_dogs = norm.fit(dogs_centroid)
mu_goats, std_goats = norm.fit(goats_centroid)

# Plot
x_cats = np.linspace(min(cats_centroid), max(cats_centroid), 150)
p_cats = norm.pdf(x_cats, mu_cats, std_cats)
x_dogs = np.linspace(min(dogs_centroid), max(dogs_centroid), 150)
p_dogs = norm.pdf(x_dogs, mu_dogs, std_dogs)
x_goats = np.linspace(min(goats_centroid), max(goats_centroid), 150)
p_goats = norm.pdf(x_goats, mu_goats, std_goats)
plt.figure()
plt.plot(x_cats, p_cats, 'k', linewidth=2, label="cats")
plt.plot(x_dogs, p_dogs, 'r', linewidth=2, label="dogs")
plt.plot(x_goats, p_goats, 'b', linewidth=2, label="goats")
plt.title(f'Delta Spectral Centroid')
plt.xlabel('Delta Spectral Centroid Mean Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# Spectral Bnadwidth
cats_bandwidth = np.array(cat_features[:, 39])
dogs_bandwidth = np.array(dog_features[:, 39])
goats_bandwidth = np.array(goat_features[:, 39])

# Fit Gaussian
mu_cats, std_cats = norm.fit(cats_bandwidth)
mu_dogs, std_dogs = norm.fit(dogs_bandwidth)
mu_goats, std_goats = norm.fit(goats_bandwidth)

# Plot
x_cats = np.linspace(min(cats_bandwidth), max(cats_bandwidth), 150)
p_cats = norm.pdf(x_cats, mu_cats, std_cats)
x_dogs = np.linspace(min(dogs_bandwidth), max(dogs_bandwidth), 150)
p_dogs = norm.pdf(x_dogs, mu_dogs, std_dogs)
x_goats = np.linspace(min(goats_bandwidth), max(goats_bandwidth), 150)
p_goats = norm.pdf(x_goats, mu_goats, std_goats)

plt.figure()
plt.plot(x_cats, p_cats, 'k', linewidth=2, label="cats")
plt.plot(x_dogs, p_dogs, 'r', linewidth=2, label="dogs")
plt.plot(x_goats, p_goats, 'b', linewidth=2, label="goats")
plt.title('Spectral Bandwidth')
plt.xlabel('Spectral Bandwidth Mean Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# Zero Crossing
cats_cross_rate = np.array(cat_features[:, 40])
dogs_cross_rate = np.array(dog_features[:, 40])
goats_cross_rate = np.array(goat_features[:, 40])

# Fit Gaussian
mu_cats, std_cats = norm.fit(cats_cross_rate)
mu_dogs, std_dogs = norm.fit(dogs_cross_rate)
mu_goats, std_goats = norm.fit(goats_cross_rate)

# Plot
x_cats = np.linspace(min(cats_cross_rate), max(cats_cross_rate), 150)
p_cats = norm.pdf(x_cats, mu_cats, std_cats)
x_dogs = np.linspace(min(dogs_cross_rate), max(dogs_cross_rate), 150)
p_dogs = norm.pdf(x_dogs, mu_dogs, std_dogs)
x_goats = np.linspace(min(goats_cross_rate), max(goats_cross_rate), 150)
p_goats = norm.pdf(x_goats, mu_goats, std_goats)

plt.figure()
plt.plot(x_cats, p_cats, 'k', linewidth=2, label="cats")
plt.plot(x_dogs, p_dogs, 'r', linewidth=2, label="dogs")
plt.plot(x_goats, p_goats, 'b', linewidth=2, label="goats")
plt.title('Zero Crossing Rate')
plt.xlabel('Zero Crossing Rate Mean Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# RMS Energy
cats_rms = np.array(cat_features[:, 41])
dogs_rms = np.array(dog_features[:, 41])
goats_rms = np.array(goat_features[:, 41])

# Fit Gaussian
mu_cats, std_cats = norm.fit(cats_rms)
mu_dogs, std_dogs = norm.fit(dogs_rms)
mu_goats, std_goats = norm.fit(goats_rms)

# Plot
x_cats = np.linspace(min(cats_rms), max(cats_rms), 100)
p_cats = norm.pdf(x_cats, mu_cats, std_cats)
x_dogs = np.linspace(min(dogs_rms), max(dogs_rms), 100)
p_dogs = norm.pdf(x_dogs, mu_dogs, std_dogs)
x_goats = np.linspace(min(goats_rms), max(goats_rms), 100)
p_goats = norm.pdf(x_goats, mu_goats, std_goats)

plt.figure()
plt.plot(x_cats, p_cats, 'k', linewidth=2, label="cats")
plt.plot(x_dogs, p_dogs, 'r', linewidth=2, label="dogs")
plt.plot(x_goats, p_goats, 'b', linewidth=2, label="goats")
plt.title('RMS Energy')
plt.xlabel('RMS Energy Mean Value')
plt.ylabel('Density')
plt.legend()
plt.show()