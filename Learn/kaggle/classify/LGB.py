# encoding:utf-8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
# kaggle--Santander Customer Transaction Prediction--Santander EDA and Prediction

import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print("a:", train_df.shape)
idx = features = train_df.columns.values[2:202]
# for df in [test_df, train_df]:
#     df['sum'] = df[idx].sum(axis=1)
#     df['min'] = df[idx].min(axis=1)
#     df['max'] = df[idx].max(axis=1)
#     df['mean'] = df[idx].mean(axis=1)
#     df['std'] = df[idx].std(axis=1)
#     df['skew'] = df[idx].skew(axis=1)
#     df['kurt'] = df[idx].kurtosis(axis=1)
#     df['med'] = df[idx].median(axis=1)

# train_df[train_df.columns[202:]].head()
# test_df[test_df.columns[201:]].head()
'''
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
for feature in features:
train_df['r2_'+feature] = np.round(train_df[feature], 2)
test_df['r2_'+feature] = np.round(test_df[feature], 2)
train_df['r1_'+feature] = np.round(train_df[feature], 1)
test_df['r1_'+feature] = np.round(test_df[feature], 1)
'''
print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']

param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.38, 'boost_from_average': 'false',
    'boost': 'gbdt', 'feature_fraction': 0.04, 'learning_rate': 0.0085,
    'max_depth': -1, 'metric': 'auc', 'min_data_in_leaf': 80, 'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13, 'num_threads': 8, 'tree_learner': 'serial', 'objective': 'binary',
    'reg_alpha': 0.1302650970728192, 'reg_lambda': 0.3603427518866501, 'verbosity': 1
}
# StratifiedKFold和KFold的区别在于，StratifiedKFold会分层抽样，保证训练集，测试集中各类别样本的比例与原始数据集中相同。
# 例如target种类有2种，0和1，StratifiedKFold保证在训练集种，0和1数量相同，验证集种0和1数量也相同
# 当各类样本数量不均衡时，交叉验证对分类任务要采用StratifiedKFold，即在每折采样时根据各类样本按比例采样，
# 交叉验证的代码中缺省的就是StratifiedKFold
folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=99999)
# np.zeros生成一个一维数组array[0,0,0,......]
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
# fold_表示第几次循环（折叠），trn_idx和val_idx表示，拆分train_df之后，得到的训练集和验证集在train_df中的下标，在train_df中用iloc定位
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    # 训练集trn_data，验证集val_data
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    # 模型训练，数据集是trn_data，验证集是trn_data, val_data两个
    clf = lgb.train(param, trn_data, 1000000, valid_sets=[trn_data, val_data], verbose_eval=5000,
                early_stopping_rounds=2000)

    # 在这一轮中，用trn_data训练好的模型预测验证集val_idx，将这部分结果，保存到oof[val_idx]（oof的1/k部分）。k次循环后，oof填满
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    # 在这一轮中，用trn_data训练好的模型预测测试集test_df，将预测的结果/k，最后将k次预测的结果相加
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

    # 打印在交叉验证中，用训练集分出来的各个训练集训练的模型，在验证集上测试得到的roc_auc分数
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = predictions
sub.to_csv("submission.csv", index=False)
