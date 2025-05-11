# -*- coding: utf-8 -*-
"""
Created on Fri May  9 21:02:28 2025

@author: hp
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Tuning C for a linear SVM, using K-fold cross validation
def K_fold_cross_validation(data, labels, k, Cs):
    size = len(data)//k
    l = len(Cs)
    accuracies = []
    best_acc = 0
    
    for j in range(l):
        svm = SVC(kernel='linear', decision_function_shape='ovr', C=Cs[j]) # 'ovr' is One-vs-Rest
        for i in range(k):
            # Generating folds
            train_data = np.concatenate((data[:i * size], data[(i+1) * size:]))
            train_labels = np.concatenate((labels[:i * size], labels[(i+1) * size:]))
            valid_data = data[i * size: (i+1) * size]
            valid_labels = labels[i * size: (i+1) * size]
            
            # Normalizing the data
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            valid_data = scaler.transform(valid_data)
            
            # Train the SVM
            svm.fit(train_data, train_labels)
            
            # Predict on the test set
            valid_pred = svm.predict(valid_data)
            
            # Evaluating accuracy
            valid_acc = accuracy_score(valid_labels, valid_pred)
            accuracies.append(valid_acc)
        print(accuracies)
        avg_acc = sum(accuracies)/k
        
        if (avg_acc > best_acc):
            best_C = Cs[j]
            best_acc = avg_acc
        accuracies = []
    return best_acc, best_C

data = np.load('../Dataset/audio_feature.npz')

train_data   = data['train_data']    
train_labels = data['train_labels']  

# Best features selected
keep_cols = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 18, 35, 39, 40]

train_data = train_data[:, keep_cols]
Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

best_acc, best_C = K_fold_cross_validation(train_data, train_labels, 6, Cs)

print(f'The best C to use for the model is {best_C}')
print(f'It gives the greatest average accuracy of {best_acc * 100}')