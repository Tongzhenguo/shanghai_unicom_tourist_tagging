# coding=utf-8
import gc
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
'''
离散特征，增加模型稳定性
数值类特征取对应的排名值,对排名特征作等值离散化
'''

def trans_data( tr_x,isTrain=True ):
    print('开始构造训练集---------------------------------------------------')
    name = '' if isTrain else '_te'
    path = '/data/cache/xgb_discretization{0}.csv'.format(name)
    if os.path.exists(path):
        return pd.read_csv( path )
    else:
        ######################################################### fill nan ########################################################
        drop_list = ['用户标识','性别','年龄段','手机品牌','手机终端型号','是否有跨省行为','是否有出境行为'
                     ,'漫入省份','漫出省份']
        tr_x = tr_x.drop(drop_list, axis=1)
        gc.collect()
        idx_name_map = pd.DataFrame( tr_x.columns,columns=['col_name'] )
        idx_name_map.index = range( len(tr_x.columns) )
        idx_name_map = idx_name_map['col_name'].to_dict()
        #排序特征
        for i in range( len(tr_x.columns) ):
            tr_x['{0}_rn'.format( idx_name_map[i] )] = tr_x[idx_name_map[i]].rank( method='max' )
            del tr_x[idx_name_map[i]]
        gc.collect()
        linspace = np.linspace(0, len(tr_x), 11)
        for i in range( 1,11 ):
            tr_x[ (tr_x>linspace[i-1]) & ( tr_x<=linspace[i] )] = i
        # 离散特征的命名：在排序特征后加'd',如'f1_rn'的离散特征为'f1d'
        rename_dict = {s: s[:-2]+'d' for s in tr_x.columns.tolist()}
        tr_x = tr_x.rename(columns=rename_dict)
        tr_x.to_csv( path,index=False )
        return tr_x

def xgb_eval(  ):
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    # del tr_x['用户标识']
    tr_x = trans_data( tr_x )
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼']
    X_train, X_test, y_train, y_test = train_test_split( tr_x,y,test_size=0.3,random_state=20170805 )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    for n in [ 100,500,1000,2000 ]:
        for d in [5,8,11]:
            param = {'max_depth': d #过拟合
                , 'min_child_weight': 25 #以前是5，过拟合
                , 'gamma': 0.1
                , 'subsample': 0.7
                , 'colsample_bytree': 0.6
                , 'eta': 0.01
                , 'lambda': 250  # L2惩罚系数，过拟合
                ,'scale_pos_weight':763820 / 81164 #处理正负样本不平衡，
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
            bst.save_model('/data/model/xgb_discretization_{0}_{1}.model'.format( n,d ))
            print
            print
# xgb_eval()

def sub_xgb( n,d ):
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    # del tr_x['用户标识']
    tr_x = trans_data(tr_x)
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
    clf.save_model('/data/model/xgb_discretization_{0}_{1}.model'.format(n, d))

    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model('/data/model/xgb_discretization_{0}_{1}.model'.format(n, d))  # load data
    sub = pd.DataFrame()
    te_x = pd.read_csv('/data/q1/data_test.csv')
    sub['IMEI'] = te_x['用户标识']
    te_X = trans_data(te_x,False)
    # del te_X['用户标识']
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
    s.to_csv('/data/cache/xgb_discretization_feature_scores.csv',index=False)
    te_X = xgb.DMatrix(te_X)
    sub['SCORE'] = clf.predict(te_X)
    sub.to_csv('/data/res/result.csv', index=False)
sub_xgb(500,8)




