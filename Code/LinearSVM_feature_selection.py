# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:11:39 2025

@author: hp
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Selecting useful features for a linear SVM, using K-fold cross validation
def K_fold_cross_validation(data, labels, k, keep_features):
    size = len(data)//k
    l = len(keep_features)
    accuracies = []
    best_acc = 0
    
    for j in range(l):
        svm = SVC(kernel='linear', decision_function_shape='ovr') # 'ovr' is One-vs-Rest
        for i in range(k):
            # Generating folds
            train_data = np.concatenate((data[:i * size], data[(i+1) * size:]))
            train_labels = np.concatenate((labels[:i * size], labels[(i+1) * size:]))
            valid_data = data[i * size: (i+1) * size]
            valid_labels = labels[i * size: (i+1) * size]
            
            # Selecting features
            train_data = train_data[:, keep_features[j]]
            valid_data = valid_data[:, keep_features[j]]
            
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            valid_data = scaler.transform(valid_data)
            
            # Train the SVM
            svm.fit(train_data, train_labels)
            
            # Predict on the test set
            valid_pred = svm.predict(valid_data)
        
            valid_acc = accuracy_score(valid_labels, valid_pred)
            accuracies.append(valid_acc)
        print(accuracies)
        avg_acc = sum(accuracies)/k
        
        if (avg_acc > best_acc):
            best_features = keep_features[j]
            best_acc = avg_acc
        accuracies = []
    return best_acc, best_features

data = np.load('../Dataset/audio_feature.npz')

train_data   = data['train_data']    
train_labels = data['train_labels']  
# test_data    = data['test_data']    
# test_labels  = data['test_labels']

keep_cols = [[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 18, 35, 39, 40],
             [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 20, 35, 39, 40],
             [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 20, 29, 35, 39],
             [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 20, 25, 29, 35, 39],
             [1, 2, 3, 5, 6, 8, 9, 10, 11, 14, 20, 25, 29, 35, 39],
             [1, 2, 3, 5, 6, 8, 9, 10, 11, 14, 22, 25, 29, 35, 39],
             [1, 2, 3, 5, 6, 8, 9, 10, 12, 14, 22, 25, 29, 35, 39]]


best_acc, best_features = K_fold_cross_validation(train_data, train_labels, 6, keep_cols)

print(f'The best features to use for the model are {best_features}')
print(f'It gives the greatest average accuracy of {best_acc * 100}')

