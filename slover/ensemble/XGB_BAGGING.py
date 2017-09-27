# coding=utf-8
import gc
import os
import sys

import pandas as pd
import xgboost as xgb

reload(sys)
sys.setdefaultencoding('utf8')
'''
各模型刷选出topk的特征，然后模型自融合
'''
def trans_data( isTrain=True ):
    print('========================开始组合特征===============================')
    raw_data_feat = list(pd.read_csv('/data/cache/xgb_feature_scores.csv').head(250)['feature'].values)
    discretization_feat = list(pd.read_csv('/data/cache/xgb_discretization_feature_scores.csv')['feature'].values)
    ratio_feat = list(pd.read_csv('/data/cache/xgb_ratio_feature_scores.csv').head(35)['feature'].values)
    combine_feat = list(pd.read_csv('/data/cache/xgb_combine_feature_scores.csv').head(70)['feature'].values)
    cross_feat = list(pd.read_csv('/data/cache/xgb_cross_feature_scores.csv').head(70)['feature'].values)
    name = ''
    if not isTrain:
        name = '_te'
    path = '/data/cache/xgb_bagging_v4{0}.csv'.format( name )
    if os.path.exists( path ):
        return pd.read_csv( path )
    else:
        raw_data = pd.read_pickle('/data/cache/xgb{0}.pkl'.format(name))[ raw_data_feat ]
        freq_data = pd.read_pickle('/data/cache/model_freq{0}.pkl'.format(name))['freq']
        discretization_data = pd.read_csv('/data/cache/xgb_discretization{0}.csv'.format(name))[ discretization_feat ]
        discretization_n_data = pd.read_pickle('/data/cache/xgb_discretization_n{0}.pkl'.format(name))
        ratio_data = pd.read_pickle('/data/cache/xgb_ratio{0}.pkl'.format(name))[ ratio_feat ]
        combine_data = pd.read_csv('/data/cache/xgb_combine{0}.csv'.format(name))[ combine_feat ]
        cross_data = pd.read_pickle('/data/cache/xgb_cross{0}.pkl'.format(name))[cross_feat]
        tr_x = pd.concat( [raw_data,freq_data,discretization_data,discretization_n_data,ratio_data,combine_data,cross_data],axis=1,join='inner' )
        print tr_x.head()
        print len(tr_x)
        tr_x.to_csv( path,index=False )
        return tr_x

def sub_xgb( gamma,n ):
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼']
    tr_x = trans_data()
    dtrain = xgb.DMatrix(tr_x, label=y)
    param = {'max_depth': 12
        , 'min_child_weight': 25  # 以前是5,过拟合
        , 'gamma': gamma
        , 'subsample': 1
        , 'colsample_bytree':1
        , 'eta': 0.01
        , 'lambda': 750  # L2惩罚系数,过拟合
        , 'scale_pos_weight': 763820 / 81164  # 处理正负样本不平衡,
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 100  # eval 得分没有继续优化 就停止了
        , 'seed': 2000
        , 'nthread': 2
        , 'silent': 0
        }
    clf = xgb.train(param, dtrain, num_boost_round=n)
    clf.save_model('/data/model/xgb_v4_{0}_{1}.model'.format(gamma,n))
    clf = xgb.Booster({'nthread': 2})  # init model
    clf.load_model('/data/model/xgb_v4_{0}_{1}.model'.format(gamma,n))
    sub = pd.DataFrame()
    # te_x = pd.read_csv('/data/q1/data_test.csv')
    # sub['IMEI'] = te_x['用户标识']
    # del te_x
    # gc.collect()
    te_X = trans_data(False)
    sub['IMEI'] = te_X['用户标识']
    te_X = xgb.DMatrix(te_X)
    sub['SCORE'] = clf.predict(te_X)
    sub.to_csv('/data/res/result_xgb_v4_{0}_{1}.csv'.format(gamma,n), index=False)

def xgb_bagging(  ):
    sub = pd.DataFrame()
    for gamma in [0.3]:
        for n in [1900]: #1900过拟合
            sub_xgb(gamma,n)
    for gamma in [0.3]:
        for n in [1900]:
            res = pd.read_csv('/data/res/result_xgb_v4_{0}_{1}.csv'.format(gamma,n))
            sub = sub.append( res )
    sub = sub.groupby(['IMEI'],as_index=False).mean()
    sub.to_csv('/data/res/ensemble/xgb_v4_bagging.csv',index=False)
xgb_bagging()
