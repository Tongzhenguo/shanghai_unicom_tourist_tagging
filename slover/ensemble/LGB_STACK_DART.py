# coding=utf-8
import os
import gc
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import KFold,ParameterGrid
from sklearn import cross_validation,metrics
import lightgbm as lgb

def trans_data( isTrain=True ):
    print('========================开始组合特征===============================')
    raw_data_feat = list(pd.read_csv('/data/cache/xgb_feature_scores.csv').head(250)['feature'].values)
    discretization_feat = list(pd.read_csv('/data/cache/xgb_discretization_feature_scores.csv')['feature'].values)
    ratio_feat = list(pd.read_csv('/data/cache/xgb_ratio_feature_scores.csv').head(35)['feature'].values)
    # combine_feat = list(pd.read_csv('/data/cache/xgb_combine_feature_scores.csv').head(70)['feature'].values)
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
        combine_data = pd.read_csv('/data/cache/xgb_combine{0}.csv'.format(name)).ix[ :,:50 ]
        cross_data = pd.read_pickle('/data/cache/xgb_cross{0}.pkl'.format(name))[cross_feat]
        tr_x = pd.concat( [raw_data,freq_data,discretization_data,discretization_n_data,ratio_data,combine_data,cross_data],axis=1,join='inner' )
        print( tr_x.head() )
        print( len(tr_x) )
        tr_x.to_csv( path,index=False )
        return tr_x

kf = KFold( n_splits=3,random_state=20170822,shuffle=True )
n_fold = 1
train = trans_data()
test = trans_data(False)
sub = pd.DataFrame()
sub['IMEI'] = test['用户标识']
label = pd.read_csv('/data/unicom_disney/data/q1/label.csv')
y = label['是否去过迪士尼'].values
del label
gc.collect()
X_train = train.values
X_test = test.values
del test
gc.collect()

gridParams={
    '4_boosting_type':['dart']
    , '2_num_leaves':[1024]
    , '1_max_depth':[9]
    , '3_min_data_in_leaf':[150]
}
max_mean_auc=0
param_list = ParameterGrid(gridParams)
print("total param comp is ", len(param_list))
for param in param_list:
    lgb_params = {
        'task': 'train',
        'boosting_type': param['4_boosting_type'],
        'num_leaves':param['2_num_leaves'],#调参,，过拟合
        'max_depth':param['1_max_depth'],
        'min_data_in_leaf':param['3_min_data_in_leaf'],#调参,，过拟合
        'metric': 'auc',
        'early_stopping_rounds': 50,
        'objective': 'binary',
    }
    mean_auc = 0
    i = 0
    if n_fold > 0:
        print('{0} flod,Start training...'.format(i))
        kf_X_train, kf_X_test, kf_y_train, kf_y_test = cross_validation.train_test_split(X_train, y, test_size=0.2, random_state=0)
        print("{0} fold: train size = {1}, P size = {2}; test size = {3}, P size = {4}".format(i, len(kf_y_train), np.sum(kf_y_train==1), len(kf_y_test), np.sum(kf_y_test==1)))
        lgb_train = lgb.Dataset(kf_X_train,label=kf_y_train,params=lgb_params)
        lgb_eval = lgb.Dataset(kf_X_test, kf_y_test, reference=lgb_train)
        clf = lgb.train(lgb_params,lgb_train, num_boost_round=1000,valid_sets=lgb_eval, early_stopping_rounds=50)
        print('{0} flod,End training, best iteration is {1}'.format(i, clf.best_iteration))
        oof_train_skf = clf.predict( kf_X_test, num_iteration=clf.best_iteration )
        oof_test_skf = clf.predict(X_test)
        test_auc = metrics.roc_auc_score(kf_y_test,oof_train_skf)#验证集上的auc值
        mean_auc += test_auc
        print("params: ", param,", auc: ", test_auc)
        tr_path = '/data/cache/oof_train{0}.pkl'.format( i )
        te_path = '/data/cache/oof_test{0}.pkl'.format( i )
        if os.path.exists(tr_path):
            os.remove(tr_path)
        if os.path.exists(te_path):
            os.remove(te_path)
        pd.to_pickle(oof_train_skf.reshape(-1,1), tr_path)
        pd.to_pickle(oof_test_skf.reshape(-1,1), te_path)
        del oof_train_skf
        del clf
        del lgb_eval
        del lgb_train
        del kf_X_train
        del kf_y_train
        del kf_X_test
        del kf_y_test
        gc.collect()
    mean_auc = mean_auc/n_fold
    ts = datetime.datetime.now()
    print(ts," params: ", param,", mean_auc: ", mean_auc)
    if mean_auc > max_mean_auc:
        max_mean_auc = mean_auc
        max_mean_auc_param = param
        print('max_mean_auc_param: ', param, ', auc: ', max_mean_auc)
    del lgb_params
    gc.collect()
print('---------- train done -------------')
