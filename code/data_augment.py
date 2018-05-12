# -*- coding: utf-8 -*-
"""
@author: zhaoguangjun
@date: 2018.05.11  17:04  星期五
@desc: 对样本进行过采样
"""
import pandas as pd
import unittest
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class KNNOverSampling:
    """ 类KNNOverSampling对少类样本进行过采样: 对于一个少量样本x, 先求其周边最近的K个样本, 如果
    有多余floor(K/2)个样本同样属于少类样本, 则分别使用这几个样本与x之间生成新的样本. 原始特征都
    是整数值, 为了计算最近距离, 事先进行了归一化处理. 生成的新样本需要进行反变换处理才行.
    
    Functions list:
    ---------------
    - normalization(self, X): 对输入样本进行归一化处理;
    - knn(self, x, dataset, k = 5): 找到样本最近的k个样本
    - generate_sample(self, sample1, sample2): 生成一个新样本
    - fit(self, X, y): 进行过采样
    """
    
    def __init__(self, sample_label = 1, k=5):
        """ 
        - sample_label: 代采样样本的标签
        - k: 进行knn的k值
        """
        self.sample_label = sample_label
        self.k = k
    
    def fit(self, X: np.array, y: np.array):
        """ 执行过采样过程
        """
        # 先对每个特征分量进行归一化处理
        X, scalers = self.normalization(X.copy())
        
        # 对少量样本进行过采样
        samples = np.empty((X.shape[1],))
        sampling_idx = np.squeeze(np.argwhere(y==self.sample_label))
        
        cnt = 0
        for i in tqdm(sampling_idx[:6000]): 
            index = KNNOverSampling.knn(X[i,:], X, k=self.k)
            index_label = y.iloc[index].values
            if sum(index_label) > np.floor(self.k/2):
                cnt += 1
                for j in range(len(index_label)):
                    if index_label[j] == 1:
                        for k in range(2):
                            new_sample = self.generate_sample(X[i,:], X[index[j],:])
                            samples    = np.vstack((samples, new_sample))
                         
        print('cnt', cnt)
        
        np.savez_compressed('tmp', samples)
        
        # 对新样本进行反归一化
        samples = scalers[1].inverse_transform(samples)
        samples = scalers[0].inverse_transform(samples)
        samples = np.round(samples)    
        return samples


    def normalization(self, X):
        """ 对样本进行归一化处理: 标准归一化 + 最小最大归一化
        """
        columns = X.columns
        for col_name in columns:
            X[col_name] = X[col_name].fillna(X[col_name].mode().values[0])
        
        ss_scaler = StandardScaler()
        mm_scaler = MinMaxScaler()
        new_X = ss_scaler.fit_transform(X)
        new_X = mm_scaler.fit_transform(new_X)
        return new_X, [ss_scaler, mm_scaler]
        
    @staticmethod      
    def knn(x: np.array, dataset: np.array, k = 5):
        """ 从samples中找到与x最近的k个索引
        
        Output:
        -------
        index: 最相似的样本所在的索引
        """
        dist = (np.subtract(x, dataset) ** 2).sum(axis=1)
        index = dist.argsort()[1:(k+1)]
        return index

        
    def generate_sample(self, sample1: np.array, sample2: np.array) -> np.array:
        """ 在sample1和sample2之间以生成随机数的方式生成一个新样本
        Input:
        ------
        sample1: 样本1
        sample2: 样本2
        
        Output:
        ------
        new_sample: 新样本
        """
        assert len(sample1) == len(sample2)
        if np.sum(sample1 - sample2) == 0:
            return None
        
        n = len(sample1)
        rand = np.random.rand(n)
        new_sample = (sample2 - sample1) * rand + sample1
        return new_sample


#class TestKNNOverSampling(unittest.TestCase):
#    def setUp(self):
#        self.p = KNNOverSampling(sample_label = 1, k=5)
#        
#    def test_generate_sample(self):
#        self.assertEqual((4,), self.p.generate_sample(sample1=np.array([1,9,56,2]),
#                                                      sample2=np.array([3,9,6,5])).shape)
#        
#    def test_normalization(self):
#        self.assertEquals([0,0,0,0], self.p.normalization(X=np.array([[1,6,5,8],[5,9,3,4],[6,8,7,5]]))[0].min())
        
        
if __name__ == '__main__':
#    unittest.main()
#    train = pd.read_csv('../data/atec_anti_fraud_train.zip', compression='zip', parse_dates=['date'])
#    train = train[train['label'] != -1]
    
#    variable_list = ['f'+str(i) for i in range(1,298)]

    # 
    print('perform over sampling...')
    kos = KNNOverSampling(k=7)
    samples = kos.fit(X=train[variable_list], y=train['label'])
































