# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:49:01 2018

@author: zbj
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve 
import numpy as np
import pandas as pd
from contextlib import contextmanager
import time
from time import strftime
from tqdm import tqdm

def mayi_score(y_true, y_score): 
    """ Evaluation metric
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label = 1) 
    score = 0.4 * tpr[np.where(fpr>=0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr>=0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr>=0.01)[0][0]] 
            
    return score

@contextmanager
def timer(message: str):
    """ Time counting
    """
    print('[{}][{}] Begin ...'.format(strftime('%Y-%m-%d %H:%M:%S'), message))
    t0 = time.time()
    yield
    print('[{}][{}] End   ...'.format(strftime('%Y-%m-%d %H:%M:%S'), message))
    print('[{}][{}] Cost {:.2f} s'.format(strftime('%Y-%m-%d %H:%M:%S'), message, time.time()-t0))


def preprocess(data: pd.DataFrame):
    """ 对数据进行预处理
    """
    def fill_outliers(col: pd.Series):
        """ Remove outliers of each col
        """
        mean, std = col.mean(), col.std()
        upper, lower = mean + 3 * std, mean - 3 * std
        col[col>upper] = np.floor(upper)
        col[col<lower] = np.floor(lower)
        return col.values
    
    # 处理离散值 & 填充空值(使用众数填充)
    columns = data.columns
    for col_name in tqdm(columns):
        data[col_name] = fill_outliers(data[col_name].copy())
        mode = data[col_name].mode().values[0]
        data[col_name] = data[col_name].fillna(mode)

    return data


def normalization(X):
    """ 对样本进行归一化处理: 标准归一化 + 最小最大归一化
    """
    ss_scaler = StandardScaler()
    mm_scaler = MinMaxScaler()
    new_X = ss_scaler.fit_transform(X)
    new_X = mm_scaler.fit_transform(new_X)
    
    return new_X, [ss_scaler, mm_scaler]




