# -*- coding: utf-8 -*-
"""
@date: Created on Sun May 13 22:37:21 2018
@author: zhaoguangjun
@desc: 训练lightgbm分类器
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from mayi import mayi_score, timer, preprocess


def evaluate(y_true, y_pred, y_prob):
    """ 估计结果: precision, recall, f1, auc, mayi_score
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mayi = mayi_score(y_true, y_prob)
    
    return [p,r,f1,auc,mayi]    
    

def kfold_model_train(clf_fit_params, clf, X, y, n_splits=5):
    """ 进行K-fold模型训练
    """
    models, i = [], 0
    eval_train = pd.DataFrame(index=range(n_splits), columns=['P','R','F1','AUC','mayi'])
    eval_test  = pd.DataFrame(index=range(n_splits), columns=['P','R','F1','AUC','mayi'])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(kf.split(X)):
        X_, X_test = X[train_index], X[test_index]
        y_, y_test = y[train_index], y[test_index]        
        
        # Divide into train and validation set for early-stop
        X_train, X_valid, y_train, y_valid = train_test_split(X_, y_, test_size=0.15, random_state=42)
        
        del X_, y_
        gc.collect()
        
        # Model Training
        clf.fit(X=X_train, y=y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc',
                verbose=False, **clf_fit_params)
        
        ## Model Testing
        # On training set
        y_prob_train = clf.predict_proba(X_train, num_iteration=clf.best_iteration)[:,1]
        y_pred_train = clf.predict(X_train, num_iteration=clf.best_iteration)
        eval_train.iloc[i,:] = evaluate(y_train, y_pred_train, y_prob_train)
        
        # On testing set
        y_prob_test = clf.predict_proba(X_test, num_iteration=clf.best_iteration)[:,1]
        y_pred_test = clf.predict(X_test, num_iteration=clf.best_iteration)
        eval_test.iloc[i,:] = evaluate(y_test, y_pred_test, y_prob_test)
        
        # Saving model
        models.append(clf)
        i += 1
        
    return models, eval_train, eval_test
        
#%%    
if __name__ == '__main__':
    # 加载数据
    with timer('Load Data'):
        train = pd.read_csv('../data/SMOTE_samples_original.zip', compression='zip')
        
    with timer('Model Train'):
        feature_name = ['f'+str(i) for i in range(1,298)] # 所有变量的名称
        nunique = train[feature_name].nunique()  # 每个特征分量unique值的数量
        categorical_feature = list(nunique[nunique <= 10].index.values) # 所有类别变量的名称
        
        # 训练样本以及类别标签
        X, y = train[feature_name].values, train['label'].values
        
        # 构造分类器
        lgb_params = {'boosting_type': 'gbdt',
                      'num_leaves': 31,
                      'max_depth': 10,
                      'learning_rate': 0.10,
                      'n_estimators': 100,
                      'reg_alpha': 0.1,
                      'seed': 42,
                      'nthread': -1}
        
        clf = lgb.LGBMClassifier(**lgb_params)
        
        # 分类器训练
        clf_fit_params = {'early_stopping_rounds': 5, 'feature_name': feature_name,
                          'categorical_feature': categorical_feature}
        
        models, eval_train, eval_test = kfold_model_train(clf_fit_params, clf, X, y, n_splits=5)

    with timer('Mode test'):
        test = pd.read_csv('../data/atec_anti_fraud_test_a.zip', compression='zip', parse_dates=['date'])
        X_test = preprocess(test[feature_name].copy()) 
                
        test_prob_final = np.zeros((len(X_test),))
        for model in models:
            test_prob = model.predict_proba(X_test, num_iteration=clf.best_iteration)[:,1]
            test_prob_final += (test_prob*0.2)

    with timer('Write Result'):
        result = pd.DataFrame()
        result['id'] = test['id']
        result['score'] = test_prob_final
        result.to_csv('../submission/submission_180514_v4.csv', index=False)

        result['pred'] = (result['score'] > 0.5).astype('int')





