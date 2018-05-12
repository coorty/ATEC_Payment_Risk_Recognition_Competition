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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_curve 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

from operator import itemgetter
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import time
from time import strftime
from contextlib import contextmanager

@contextmanager
def timer(message: str):
    """ Time counting
    """
    print('[{}][{}] Begin ...'.format(strftime('%Y-%m-%d %H:%M:%S')), message)
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

def preprocess(data: pd.DataFrame):
    """ 对数据进行预处理
    """
    def fill_outliers(col: pd.Series):
        """ Remove outliers of each col
        """
        mean = col.mean()
        std  = col.std()
        upper = mean + 3 * std
        lower = mean - 3 * std
        col[col>upper] = np.floor(upper)
        col[col<lower] = np.floor(lower)
        return col.values
    
    # 处理离散值 & 填充空值(使用众数填充)
    columns = data.columns
    for col_name in tqdm(columns):
        data[col_name] = fill_outliers(data[col_name].copy())
        mode = data[col_name].mode().values[0]
        data[col_name] = data[col_name].fillna(mode).astype('float64')

    return data

def normalization(X):
    """ 对样本进行归一化处理: 标准归一化 + 最小最大归一化
    """
    ss_scaler = StandardScaler()
    mm_scaler = MinMaxScaler()
    new_X = ss_scaler.fit_transform(X)
    new_X = mm_scaler.fit_transform(new_X)
    
    return new_X, [ss_scaler, mm_scaler] 

def make_field_pipeline(field: str, *vec) -> Pipeline:
    """ Make Pipeline with refer to field : `field`, and some transform functions: `*vec`
    Input:
        - field: a data field
        - *vec: a sequence transformance functions
    """
    return make_pipeline(FunctionTransformer(itemgetter(field), validate=False), *vec)

if __name__ == '__main__':
    with timer('Load data'):
        train  = pd.read_csv('../data/atec_anti_fraud_train.zip', compression='zip', parse_dates=['date'])
        train.sort_values('label', inplace=True)
        train2 = pd.read_csv('../data/Maked_Pos_Samples.csv')
        test   = pd.read_csv('../data/atec_anti_fraud_test_a.zip', compression='zip', parse_dates=['date'])
        merge  = pd.concat((train, train2, test), ignore_index=True)
        
        
    with timer('Manage variables'):
        # 所有可用的变量的名称
        all_vars = ['f'+str(i) for i in range(1,298)]
        
        # 需要做PCA处理的变量
        pca_vars_lists = []
        pca_vars_all   = []
        pca_vars_name = 'pca_variables.txt'
        with open(pca_vars_name, 'r') as f:
            for line in f:
                list_ = ['f'+str(int(ele)) for ele in line.split(',')]
                pca_vars_lists.append(list_)
                pca_vars_all += list_
        
        # 排除了需要做PCA的变量, 剩下的变量
        other_vars = [ele for ele in all_vars if ele not in pca_vars_all]    
        
    with timer('Data preprocess'):
        # 数据预处理
        merge[all_vars] = preprocess(merge[all_vars].copy())            
    
    with timer('Features Extraction'):
        # 用作特征提取
        vectorizer = make_union(
                *[make_field_pipeline(pca_vars_lists[i],
                                      PCA(n_components=int(np.ceil(len(pca_vars_lists[i])/2)))) for i in range(len(pca_vars_lists))],
                  make_field_pipeline(other_vars)
                )
        
        feats = vectorizer.fit_transform(merge[all_vars])
    
        # 分成训练和测试
        feats_train = feats[((merge['label']==1) | (merge['label']==0)).values, :]
        y_train = merge.loc[(merge['label']==1) | (merge['label']==0), 'label'].values
        feats_test  = feats[merge['label'].isnull().values, :]
        
        pos_train, neg_train = feats_train[y_train==1], feats_train[y_train==0]
        
        del merge
    
    with timer('Model train'):
        rf = RandomForestClassifier(n_estimators=20,
                                    max_depth=10,
                                    n_jobs=-1)
            
        num_pos = int(sum(y_train))             # 正样本数量
        num_neg = int(len(y_train) - num_pos)   # 负样本数量
        
        # 训练多个模型
        classify_models, classify_num = [], 10
        neg_rate = 4
        
        for i in range(classify_num):
            print('Training ', str(i), ' classifier')
    
            # Make samples
            num_pos_select = int(num_pos - 3000)
            num_neg_select = int(np.floor(num_pos_select * 2.5))  # 选择的负样本数量
        
            # 随机选出一部分正样本
            pos_flag = np.random.choice([1,0], num_pos, p=[num_pos_select/num_pos,
                                        1-num_pos_select/num_pos])
            # 随机选出一部分负样本
            neg_flag = np.random.choice([1,0], num_neg, p=[num_neg_select/num_neg,
                                        1-num_neg_select/num_neg])
    
            X = np.vstack((pos_train[pos_flag==1], neg_train[neg_flag==1]))
            y = np.squeeze(np.vstack((np.ones((sum(pos_flag), 1)), np.zeros((sum(neg_flag), 1)))))

            # Fit the model
            rf = rf.fit(X=X, y=y)
            classify_models.append(rf)
            
            ## Test on the train dataset
            y_pred_train = rf.predict(X)
            y_prob_train = rf.predict_proba(X)[:,np.where(rf.classes_==1)[0][0]]
            
            print('test score on train dataset is ', str(score(y, y_prob_train)))
            print(confusion_matrix(y, y_pred_train))
            
            ## Test on the whole dataset
            y_pred = rf.predict(feats_train)
            y_prob = rf.predict_proba(feats_train)[:,np.where(rf.classes_==1)[0][0]]
            
            print('test score on whole dataset is ', str(score(y_train, y_prob)))
            print('混淆矩阵:\n', confusion_matrix(y_train, y_pred))

    with timer('Mode test'):
        test_prob_final = np.zeros((len(feats_test),))
        for i in range(classify_num):
            test_prob = classify_models[i].predict_proba(feats_test)[:,1]
            test_prob_final += (test_prob*0.1)

        
    with timer('Write Result'):
        result = pd.DataFrame()
        result['id'] = test['id']
        result['score'] = test_prob_final
        result.to_csv('../submission/submission_180512_v2.csv', index=False)

# Load the data
# 特征之间的余弦相似度

#mat = pd.DataFrame(cosine_similarity(merge[variable_list].T), 
#                   columns=variable_list,
#                   index=variable_list)
#
#mat_pear = merge[variable_list].corr()

