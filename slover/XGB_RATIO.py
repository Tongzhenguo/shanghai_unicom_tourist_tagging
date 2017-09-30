# coding=utf-8
import gc
import os
import sys

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
'''
统计各app和app分类的pv占比，筛选出对旅游者影响较大的app和app分类
'''
reload(sys)
sys.setdefaultencoding('utf8')


def trans_data( tr_x,isTrain=True ):
    print('开始构造训练集---------------------------------------------------')
    ######################################################### fill nan #################################################################################
    path = '/data/cache/xgb_ratio.pkl'
    if not isTrain:
        path = '/data/cache/xgb_ratio_te.pkl'
    if os.path.exists( path ):
        return pd.read_pickle( path )
    else:
        drop_list = ['用户标识', '性别', '年龄段','大致消费水平', '每月的大致刷卡消费次数','手机品牌', '手机终端型号','用户更换手机频次',
                     '固定联络圈规模','是否有跨省行为', '是否有出境行为', '漫入省份', '漫出省份']
        tr_x = tr_x.drop(drop_list, axis=1)
        gc.collect()
        df = pd.DataFrame( list(tr_x.columns),columns=['name'] )
        col_name_dict = df['name'].to_dict()
        tr_x = tr_x.astype('float')
        col_cnt = tr_x.head(1).values.shape[1]
        tr_x['app_pv_sum'] = 0
        for i in range( 0,300 ):
            tr_x['app_pv_sum'] += tr_x.ix[:,i]
        tr_x['net_pv_sum'] = 0
        for i in range(300, col_cnt):
            tr_x['net_pv_sum'] += tr_x.ix[:, i]
        j = 0
        for i in range( 0,300 ):
            tr_x['app_pv_ratio_{0}'.format( col_name_dict[i] )] = tr_x.ix[:,i] / tr_x['app_pv_sum']
            del tr_x['{0}'.format(col_name_dict[i])]
            if j % 2 == 0:
                gc.collect()
            j += 1
        j = 0
        for i in range(0, 30):
            idx = 300 + i
            tr_x['net_pv_ratio_{0}'.format(col_name_dict[idx])] = tr_x.ix[:, i] / tr_x['net_pv_sum']
            del tr_x['{0}'.format(col_name_dict[idx])]
            if j % 2 == 0:
                gc.collect()
            j += 1
        gc.collect()
        pd.to_pickle( tr_x,path )
        return tr_x

def xgb_eval(  ):
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    tr_x = trans_data( tr_x )
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼']
    X_train, X_test, y_train, y_test = train_test_split( tr_x,y,test_size=0.3,random_state=20170805 )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    for n in [ 100,300,500 ]:
        for d in [ 8,9,10 ]:
            param = {'max_depth': d
                , 'min_child_weight': 15 #以前是5
                , 'gamma': 0.1
                , 'subsample': 0.7
                , 'colsample_bytree': 0.6
                , 'eta': 0.01
                , 'lambda': 750  # L2惩罚系数
                ,'scale_pos_weight':763820 / 81164 #处理正负样本不平衡,
                , 'objective': 'binary:logistic'
                , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
                , 'early_stopping_rounds': 100  # eval 得分没有继续优化 就停止了
                , 'seed': 2000
                , 'nthread': 4
                , 'silent': 1
                }
            print 'nums :{0},depth :{1}'.format( n,d )
            evallist = [(dtest, 'eval'), (dtrain, 'train')]
            bst = xgb.train(param, dtrain, num_boost_round=n, evals=evallist)
            bst.save_model('/data/model/xgb_ratio_{0}_{1}.model'.format( n,d ))
            print
            print
# xgb_eval()

def sub_xgb( n,d ):
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    tr_x = trans_data(tr_x)
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼']
    dtrain = xgb.DMatrix(tr_x, label=y)
    param = {'max_depth': d
        , 'min_child_weight': 25  # 以前是5,过拟合
        , 'gamma': 0.1
        , 'subsample': 0.7
        , 'colsample_bytree': 0.6
        , 'eta': 0.01
        , 'lambda': 250  # L2惩罚系数,过拟合
        , 'scale_pos_weight': 763820 / 81164  # 处理正负样本不平衡,
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 100  # eval 得分没有继续优化 就停止了
        , 'seed': 2000
        , 'nthread': 2
        , 'silent': 0
        }
    clf = xgb.train(param, dtrain, num_boost_round=n)
    clf.save_model('/data/model/xgb_ratio_{0}_{1}.model'.format(n, d))

    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model('/data/model/xgb_ratio_{0}_{1}.model'.format(n, d))
    sub = pd.DataFrame()
    te_x = pd.read_csv('/data/q1/data_test.csv')
    sub['IMEI'] = te_x['用户标识']
    te_X = trans_data(te_x,False)
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
    s.to_csv('/data/cache/xgb_ratio_feature_scores.csv',index=False)
    te_X = xgb.DMatrix(te_X)
    sub['SCORE'] = clf.predict(te_X)
    sub.to_csv('/data/res/result_xgb_ratio_{0}_{1}.csv'.format( n,d ), index=False)
sub_xgb(500,10)





