# coding=utf-8
import gc
import os
from optparse import OptionParser
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

parser = OptionParser()
parser.add_option("-k", help='num i ')
(options, args) = parser.parse_args()
kf_i = 0
if options.k is not None:
    kf_i = int(options.k)

def trans_data( isTrain=True ):
    name = ''
    if not isTrain:
        name = '_te'
    return pd.read_csv( '/data/cache/xgb_bagging_v4{}.csv'.format(name) )
kf = KFold( n_splits=5,random_state=666888,shuffle=True) #Seed must be between 0 and 4294967295
X_train = trans_data()
label = pd.read_csv('/data/q1/label.csv')
y = label['是否去过迪士尼']
del label
gc.collect()
param = {'max_depth': 12
        , 'min_child_weight': 55  #克服过拟合
        , 'gamma': 0.3
        , 'subsample': 1.0 #克服过拟合
        , 'colsample_bytree':0.58#克服过拟合
        , 'eta': 0.01
        , 'lambda': 750  # L2惩罚系数,过拟合
        , 'scale_pos_weight': 763820 / 81164  # 处理正负样本不平衡,适量增大正样本权重
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 30  # eval 得分没有继续优化 就停止了
        , 'seed': 20170824
        , 'nthread': 2
        , 'silent': 0
}
for i,(train_index,test_index) in enumerate( kf.split(X_train) ):
        if i != kf_i:continue
        X_test = trans_data(False)
        print('{0} flod,Start training...'.format(i))
        kf_X_train = X_train.iloc[train_index] #.iloc is deprecated. Please use .loc for label based indexing or .iloc for positional indexing
        kf_y_train = y.iloc[train_index]
        kf_X_test = X_train.iloc[test_index]
        kf_y_test = y.iloc[test_index]
        dtrain = xgb.DMatrix(kf_X_train, label=kf_y_train)
        dvalid = xgb.DMatrix(kf_X_test, label=kf_y_test)
        dtrain_kf_X_test= xgb.DMatrix(kf_X_test)
        dtest = xgb.DMatrix(X_test)
        evallist = [(dvalid, 'eval'), (dtrain, 'train')]
        del kf_X_train
        del kf_y_train
        del kf_y_test
        del kf_X_test
        del X_test
        gc.collect()
        clf = xgb.train(param, dtrain, num_boost_round=800,evals=evallist)
        print('{0} fold,End training...'.format(i))
        del dtrain
        del dvalid
        oof_train_skf = clf.predict(dtrain_kf_X_test)
        oof_test_skf = clf.predict(dtest)
        del dtrain_kf_X_test
        del dtest
        gc.collect()
        tr_path = '/data/cache/xgb_stack_oof_train{0}_v2.pkl'.format(i)
        te_path = '/data/cache/xgb_stack_oof_test{0}_v2.pkl'.format(i)
        if os.path.exists(tr_path):
            os.remove(tr_path)
        if os.path.exists(te_path):
            os.remove(te_path)
        pd.to_pickle(oof_train_skf.reshape(-1, 1), tr_path)
        pd.to_pickle(oof_test_skf.reshape(-1, 1), te_path)
