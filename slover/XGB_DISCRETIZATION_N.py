# coding=utf-8
import os

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

'''
离散特征之后的计数特征，增加模型稳定性
统计一个样本有多少取值为0,多少取值为1,...
'''

def trans_data( isTrain=True ):
    print('开始构造训练集---------------------------------------------------')
    name = '' if isTrain else '_te'
    path = '/data/cache/xgb_discretization_n{0}.pkl'.format(name)
    if os.path.exists(path):
        return pd.read_pickle( path )
    else:
        #计数特征
        tr_x = pd.read_csv('/data/cache/xgb_discretization{0}.csv'.format(name))
        tr_x['n1'] = (tr_x == 1).sum(axis=1)
        tr_x['n2'] = (tr_x == 2).sum(axis=1)
        tr_x['n3'] = (tr_x == 3).sum(axis=1)
        tr_x['n4'] = (tr_x == 4).sum(axis=1)
        tr_x['n5'] = (tr_x == 5).sum(axis=1)
        tr_x['n6'] = (tr_x == 6).sum(axis=1)
        tr_x['n7'] = (tr_x == 7).sum(axis=1)
        tr_x['n8'] = (tr_x == 8).sum(axis=1)
        tr_x['n9'] = (tr_x == 9).sum(axis=1)
        tr_x['n10'] = (tr_x == 10).sum(axis=1)
        tr_x = tr_x[['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10']]
        pd.to_pickle( tr_x, path )
        return tr_x

def sub_xgb( n,d ):
    tr_x = trans_data(  )
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼']
    dtrain = xgb.DMatrix(tr_x, label=y)
    param = {'max_depth': d  # 过拟合
        , 'min_child_weight': 25  # 以前是5，过拟合
        , 'gamma': 0.1
        , 'subsample': 0.7
        , 'colsample_bytree': 0.6
        , 'eta': 0.01
        , 'lambda': 250  # L2惩罚系数，过拟合
        , 'scale_pos_weight': 763820 / 81164  # 处理正负样本不平衡，
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 100  # eval 得分没有继续优化 就停止了
        , 'seed': 2000
        , 'nthread': 2
        , 'silent': 0
    }
    clf = xgb.train(param, dtrain, num_boost_round=n)
    clf.save_model('/data/model/xgb_discretization_n_{0}_{1}.model'.format(n, d))
    sub = pd.DataFrame()
    te_x = pd.read_csv('/data/q1/data_test.csv')
    sub['IMEI'] = te_x['用户标识']
    te_X = trans_data(False)
    feat = te_X.columns
    data = pd.DataFrame( feat,columns=['feature'] )
    data['col'] = data.index
    feature_score = clf.get_fscore()
    keys = []
    values = []
    for key in feature_score:
        keys.append( key )
        values.append( feature_score[key] )
    df = pd.DataFrame( keys,columns=['features'] )
    df['score'] = values
    df['col'] = df['features'].apply( lambda x:int(x[1:]) )
    s = pd.merge( df,data,on='col' )
    s = s.sort_values('score',ascending=False)[['feature','score']]
    s.to_csv('/data/cache/xgb_discretization_n_feature_scores.csv',index=False)
    te_X = xgb.DMatrix(te_X)
    sub['SCORE'] = clf.predict(te_X)
    sub.to_csv('/data/res/result.csv', index=False)
sub_xgb( 500,4 )




