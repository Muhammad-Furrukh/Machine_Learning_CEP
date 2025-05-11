# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:11:39 2025

@author: hp
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

data = np.load('../Dataset/audio_feature.npz')

train_data   = data['train_data']    
train_labels = data['train_labels']  
test_data    = data['test_data']    
test_labels  = data['test_labels']

keep_features = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 18, 35, 39, 40]
train_data = train_data[:, keep_features]
test_data = test_data[:, keep_features]
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

svm = SVC(kernel='linear', decision_function_shape='ovr', C=0.1)
svm.fit(train_data, train_labels)

test_pred = svm.predict(test_data)
test_acc = accuracy_score(test_labels, test_pred)
print(test_acc*100)

# Saving the trained SVM model and fitted preprocessing scaler
joblib.dump(svm, "svm.pkl")
joblib.dump(scaler, "scaler.pkl")