# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:15:13 2018
@author: zbj

对训练集中4725个未标记的样本进行打标, 最佳当的方式是使用KNN进行投票.

"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

knn = KNeighborsClassifier(n_neighbors = 5,
                           algorithm = 'kd_tree',
                           leaf_size = 30,
                           p = 2,         # 欧氏距离
                           n_jobs = -1)

print('Load data and preprocess...')

# Load data
train = pd.read_csv('../data/atec_anti_fraud_train.zip', compression='zip')

# Feature names
numer_feat_names = ['f'+str(i) for i in range(1,298)]

# Fill the Null with Mininum
for feat in numer_feat_names:
    train[feat].fillna(train[feat].min(), inplace=True)
    
# Scala
ss_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
train[numer_feat_names] = ss_scaler.fit_transform(train[numer_feat_names].values)
train[numer_feat_names] = mm_scaler.fit_transform(train[numer_feat_names].values)
   
print('KNN training ...')
knn.fit(X = train.loc[train['label'] != -1, numer_feat_names].values,
        y = train.loc[train['label'] != -1, 'label'].values)

print('KNN testing ...')
knn.predict(X = train.loc[train['label'] == -1, numer_feat_names].values)



X = train.loc[train['label'] != -1, numer_feat_names].values
y = train.loc[train['label'] != -1, 'label'].values


















