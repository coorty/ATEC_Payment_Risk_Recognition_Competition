# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:25:19 2018

@author: zbj
@url: https://dc.cloud.alipay.com/index#/topic/data?id=4

在金融行业中，风控系统与黑产的攻防几乎是无时不刻的存在着，风控系统中用来实时识别风险的机器学习
模型需要在黑产攻击的手法改变的时候能够及时的对其进行重新识别。而机器学习算法在训练过程中学习到
的黑产的攻击手法是基于历史数据中的黑样本学习而来，当模型上线后一旦黑产的攻击手法做了调整，这时
候模型的性能往往会衰退，其主要原因就是原有模型变量对黑产攻击手法的刻画已经过时了。

因此，风控中机器学习模型的一个非常重要的特性就是模型性能的时效性和稳定性要好，即把风险的变化趋势从数据的变动和
黑样本的变迁中及时的学习出来，使得模型上线后能够在尽可能长的时间内不重新训练而能持续的对黑产攻击
进行识别。在利用机器学习算法解决风控问题的过程中，另一较大的挑战是一部分样本数据的标签的缺失。
主要原因为风控系统会基于对交易的风险判断而失败掉很多高危交易，这些交易因为被失败了往往没有了标签，
而这部分数据又极其重要，因为风控系统中的黑样本量级本来就很少，一旦有了它们的标签后机器学习算法
可以对全局数据，尤其是黑样本的全貌会有更准确的了解，才能训练出可以更精准识别风险的模型。

在本次大赛中，我们将给出由一段时间内有正负标签样本的支付行为样本和没有标签的支付行为样本组成的
训练数据集和一段时间后的某个时间范围内的有正负标签的支付行为样本构成的测试数据集。同时，本次竞
赛模拟线上模型应用的真实场景，因此强调大家需要注意不要出现“数据穿越”的情况。即在对某个交易样本
生成模型变量的时候，不可以使用该交易事件点之后的样本。希望选手们通过机器学习算法和对无标签数据
的挖掘在训练集上训练出性能稳定时效性好的模型，能够在测试集上对交易的风险进行精准判断。

- id: 主键, 不会有重复;
- label: 1: 12122; 0: 977884; -1: 4725; 正负样本比例: 1.24%
- date: 交易时间


    
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.metrics import roc_curve 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

import gc
from operator import itemgetter
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import time
from time import strftime
from contextlib import contextmanager
from mayi import mayi_score

@contextmanager
def timer(message: str):
    """ Time counting
    """
    print('[{}][{}] Begin ...'.format(strftime('%Y-%m-%d %H:%M:%S'), message))
    t0 = time.time()
    yield
    print('[{}][{}] End   ...'.format(strftime('%Y-%m-%d %H:%M:%S'), message))
    print('[{}][{}] Cost {:.2f} s'.format(strftime('%Y-%m-%d %H:%M:%S'), message, time.time()-t0))

def score(y_true, y_score): 
    """ Evaluation metric
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label = 1) 
    score = 0.4 * tpr[np.where(fpr>=0.001)[0][0]] + \
            0.3 * tpr[np.where(fpr>=0.005)[0][0]] + \
            0.3 * tpr[np.where(fpr>=0.01)[0][0]] 
            
    return score 

def sample_extraction(X, y, nums, pos_index=1):
    """ 抽取正样本+负样本
    """
    num_pos = int(sum(y == pos_index))  # 总正样本数量
    num_neg = int(sum(y != pos_index))  # 总负样本数量
    
    # 随机选取一些正/负样本点
    index_pos = np.random.randint(num_pos, size=nums[0])
    index_neg = np.random.randint(num_neg, size=nums[1])
    
    pos, neg = X[y.values==pos_index].iloc[index_pos].values, X[y.values!=pos_index].iloc[index_neg].values
    
    return np.vstack((pos, neg)), np.squeeze(np.vstack((np.ones((nums[0], 1)), 
                                                        np.zeros((nums[1], 1)))))
 
    
class KSampleSubset:
    """ 用于对数据集的抽样
    """
    def __init__(self, X, y, by):
        """
        X: 特征
        y: 类别标签
        by: 分组列(如: `date`列)
        """
        self.X = X
        self.y = y
        self.by = by
        
        self.X['y']  = self.y.values
        self.X['by'] = self.by.values
        
    def sample_select(self, X, n_clusters) -> list:
        """ 使用聚类的方式提取具有代表性的样本
        
        Input:
        ------
        X: 样本集
        n_clusters: 聚类的数量
        
        Output:
        ------
        indexes: 选中的样本点在X中的索引
        """
        # 数据归一化处理
        X_norm = self.sample_norm(X.copy(), n_type='MinMax')
        
        # 随机使用3/4的样本进行聚类
        X_norm = X_norm[np.random.randint(len(X_norm), size=int(len(X_norm)*(3.0/4.0))),:] 
        
        # 进行Kmeans聚类
        mbk = MiniBatchKMeans(n_clusters=n_clusters, init_size=n_clusters+1)
        mbk.fit(X_norm)
        
        # 有了聚类中心后, 找出距离每个聚类中心最近的样本点, 返回索引
        c_centers = mbk.cluster_centers_
        
        # 找距离聚类中心最近的点
        kdtree = spatial.KDTree(X_norm)
        _, indexes = kdtree.query(c_centers)
                     
        return indexes        
    
    
    def fit_transform_v2(self, frac):
        """ 
        frac: 负样本数量与正样本数量之比
        """
        X, y = [], []
        for name, group in self.X.groupby('by'):
            """ 在每个分组中抽取一些正/负样本 """
            # 本组下的正负样本
            samples_pos = group[group['y'] == 1] # 此分组下的正样本
            samples_neg = group[group['y'] == 0] # 此分组下的负样本
            
            # 正/负样本的数量
            num_pos, num_neg = len(samples_pos), len(samples_neg)
            
            # 抽取一部分的正/负样本组成训练集
            index_pos = np.random.randint(num_pos, size=int(num_pos))      # 抽取num_pos个正样本
            index_neg = self.sample_select(X=samples_neg.iloc[:,:-2], n_clusters=int(num_pos*frac)) # 抽取num_pos*frac个负样本
                        
            # 抽取得到的正/负样本
            samples_pos_selected = samples_pos.iloc[index_pos]
            samples_neg_selected = samples_neg.iloc[index_neg]
            
            # 保存抽取到的本组内的正/负样本
            X.append(samples_pos_selected)
            y.append(np.ones((int(num_pos), 1)))
            
            X.append(samples_neg_selected)
            y.append(np.zeros((int(num_pos*frac), 1)))
            
        return np.vstack(X)[:,:-2], np.squeeze(np.vstack(y)) 
    
    
    def fit_transform(self, frac):
        """ 
        frac: 负样本数量与正样本数量之比
        """
        X, y = [], []
        for name, group in self.X.groupby('by'):
            """ 在每个分组中抽取一些正/负样本 """
            # 本组下的正负样本
            samples_pos = group[group['y'] == 1] # 此分组下的正样本
            samples_neg = group[group['y'] == 0] # 此分组下的负样本
            
            # 正/负样本的数量
            num_pos, num_neg = len(samples_pos), len(samples_neg)
            
            # 抽取一部分的正/负样本组成训练集
            index_pos = np.random.randint(num_pos, size=int(num_pos))      # 抽取num_pos个正样本
            index_neg = np.random.randint(num_neg, size=int(num_pos*frac)) # 抽取num_pos*frac个负样本
            
            # 抽取得到的正/负样本
            samples_pos_selected = samples_pos.iloc[index_pos]
            samples_neg_selected = samples_neg.iloc[index_neg]
            
            # 保存抽取到的本组内的正/负样本
            X.append(samples_pos_selected)
            y.append(np.ones((int(num_pos), 1)))
            
            X.append(samples_neg_selected)
            y.append(np.zeros((int(num_pos*frac), 1)))
            
        return np.vstack(X)[:,:-2], np.squeeze(np.vstack(y))        
        
    
    def sample_norm(self, X: pd.DataFrame, n_type='MinMax') -> pd.DataFrame:
        """ 对样本进行标准化处理(为了方便聚类算法)
        """
        if n_type == 'MinMax':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            
        elif n_type == 'Standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            return X
        
        return X
    
    
def sample_extraction2(X, y, frac, pos_index=1):
    """ 抽取正样本+负样本
    """
    num_pos = int(sum(y == pos_index))  # 总正样本数量
    num_neg = int(sum(y != pos_index))  # 总负样本数量
    
    new_X, new_y = [], []
    
    for date in X['date'].unique():
        """ 在每个组中随机选取一些正/负样本点"""
        # 随机选取一些正/负样本点
        num_pos = len(X[X['date']==date])
        num_neg = len(X[X['date']==date])
        
        index_neg = np.random.randint(num_neg, size=int(num_pos*frac))
        index_neg = np.random.randint(num_neg, size=int(num_pos*frac))
        
        pos = X[y.values==pos_index][X[date]==date].iloc[index_pos].values
        neg = X[y.values!=pos_index][X[date]==date].iloc[index_neg].values
    
        new_X.append(pos)
        new_X.append(neg)
        new_y.append(np.ones((num_pos, 1)))
        new_y.append(np.ones((int(num_pos*frac), 1)))
    
    return new_X, new_y
    

def evaluate(y_true, y_pred, y_prob):
    """ 估计结果: precision, recall, f1, auc, mayi_score
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mayi = mayi_score(y_true, y_prob)
    
    return [p,r,f1,auc,mayi] 
    
def model_train(clf, X, y):
    """ 进行模型训练
    """
    # Divide into train and validation set for early-stop
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)
        
    clf.fit(X=X_train, y=y_train)

    ## Model Testing
    # On training set
    y_prob_train = clf.predict_proba(X_train)[:,1]
    y_pred_train = clf.predict(X_train)
    eval_train = evaluate(y_train, y_pred_train, y_prob_train)
    
    
    # On testing set
    y_prob_test = clf.predict_proba(X_valid)[:,1]
    y_pred_test = clf.predict(X_valid)
    eval_test   = evaluate(y_valid, y_pred_test, y_prob_test)
        
    return clf, eval_train, eval_test
    
# Main Loop
if __name__ == '__main__':
    with timer('Load data'):
        # 加载原始训练集
        train = pd.read_csv('../data/atec_anti_fraud_train.zip', compression='zip', parse_dates=['date'])
        train = train[train['label'] != -1]  # 去除未标记的训练样本
        
        # 加载原始测试集
        test  = pd.read_csv('../data/atec_anti_fraud_test_a.zip', compression='zip', parse_dates=['date'])
        
        # 将训练集和测试集进行暂时合并       
        merge  = pd.concat((train, test), ignore_index=True)
    
    
    with timer('Data preprocess'):
        # 所有特征的名称
        all_vars = ['f'+str(i) for i in range(1,298)]
        
        # 对于每个特征的空值, 使用特征的最小值进行填充
        for var in all_vars:
            merge[var] = merge[var].fillna(merge[var].min())
    
    
    with timer('Features Extraction'):    
        # 分成训练特征集合和测试特征集合
        X_train = merge.loc[merge['label'].notnull(), all_vars]
        y_train = train['label']      # 训练集的标签
        X_train_date = merge.loc[merge['label'].notnull(), 'date'] # 单独提取训练集中日期列, 后面会用到

        # 测试集
        X_test  = merge.loc[merge['label'].isnull(), all_vars].values
        id_test = test['id'].values
        
        del merge, train, test
        gc.collect()
        
        
    with timer('Model train'):
        # 使用多个分类模型进行决策
        models = [
                RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1),
                ExtraTreesClassifier(n_estimators=20, max_depth=10, n_jobs=-1),
                #AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=10),
                LGBMClassifier(n_estimators=20, max_depth=10),
                GradientBoostingClassifier(n_estimators=15, max_depth=10),
                XGBClassifier(n_estimators=15, max_depth=8, n_jobs=-1)
                ]
        
        # 定义评价测度
        eval_train = pd.DataFrame(index=range(len(models)), columns=['P','R','F1','AUC','mayi'])
        eval_test  = pd.DataFrame(index=range(len(models)), columns=['P','R','F1','AUC','mayi'])
        
        # 开始训练
        trained_models = []
        for i in tqdm(range(len(models))):
            print('Begin to train the', str(i), '-th classifier...')
            
            # 抽取训练样本集
            kss = KSampleSubset(X=X_train.copy(), y=y_train, by=X_train_date)
            X, y = kss.fit_transform_v2(frac=3)
            
            # 保存下来
            np.savez_compressed('../samples/samples_'+str(i), X=X, y=y)
            
            # 训练模型
            clf, eval_train.iloc[i,:], eval_test.iloc[i,:] = model_train(models[i], X, y)
            trained_models.append(clf)
            

    with timer('Mode test'):        
        y_test_prob = np.zeros(shape=(len(models), len(X_test)))
        y_test_pred = np.zeros(shape=(len(models), len(X_test)))
        for i in range(len(models)):
            y_test_pred[i, :] = trained_models[i].predict(X_test) # 0/1标签
            y_test_prob[i, :] = trained_models[i].predict_proba(X_test)[:,1] # 属于正样本的概率值
            
        y_test_prob_final = np.zeros(shape=(len(X_test), 1))
        y_test_pred_sum = np.sum(y_test_pred, axis=0)
        for i, sum_ in tqdm(enumerate(y_test_pred_sum)):
            if sum_ >= 3:
                index = np.squeeze(np.argwhere(y_test_pred[:,i]==1))
            else:
                index = np.squeeze(np.argwhere(y_test_pred[:,i]==0))
            
            y_test_prob_final[i] = y_test_prob[index,i].mean()
        
        
        
    with timer('Write Result'):
        result = pd.DataFrame()
        result['id']    = id_test
        result['score'] = y_test_prob_final
        result.to_csv('../submission/submission_180517_v6.csv', index=False)

