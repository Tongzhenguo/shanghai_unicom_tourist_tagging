# coding=utf-8
import gc
import math
import os
import sys

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

reload(sys)
sys.setdefaultencoding('utf8')
'''
根据重要性分数，筛选出前k个特征,两两进行log(xy)变换
'''
def trans_data( tr_x,isTrain=True ):
    print('开始构造训练集---------------------------------------------------')
    path = '/data/cache/xgb_combine.csv'
    if not isTrain:
        path = '/data/cache/xgb_combine_te.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        s = pd.read_csv('/data/cache/xgb_feature_scores.csv')
        s = s.head(25)
        s = list(s[ (s.feature!='nnz') & (s.feature!='大致消费水平') & (s.feature!='年龄段') ]['feature'].values)
        tr_x = tr_x[ s ]
        old_list = list(tr_x.columns)
        df = pd.DataFrame(old_list, columns=['name'])
        col_name_dict = df['name'].to_dict()
        for i in range( 22 ):
            for j in range( 22 ):
                if i == j:continue
                tr_x['log_{0}_{1}'.format( col_name_dict[i],col_name_dict[j] ) ] = tr_x.ix[:,i].apply(math.log1p) + tr_x.ix[:,j].apply(math.log1p)
        tr_x = tr_x.drop( old_list,axis=1 )
        gc.collect()
        tr_x.to_csv( path,index=False )
        return tr_x

def sub_xgb( n,d ):
    tr_x = pd.read_pickle('/data/cache/xgb.pkl')
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
    clf.save_model('/data/model/xgb_combine_{0}_{1}.model'.format(n, d))

    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model('/data/model/xgb_combine_{0}_{1}.model'.format(n, d))
    sub = pd.DataFrame()
    te_x = pd.read_pickle('/data/cache/xgb_te.pkl')
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
    s.to_csv('/data/cache/xgb_combine_feature_scores.csv',index=False)
    te_X = xgb.DMatrix(te_X)
    sub['SCORE'] = clf.predict(te_X)
    sub.to_csv('/data/res/result_xgb_combine_{0}_{1}.csv'.format( n,d ), index=False)
sub_xgb(500,10)

def xgb_eval(  ):
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    tr_x = trans_data( tr_x )
    del tr_x['用户标识']
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼']
    X_train, X_test, y_train, y_test = train_test_split( tr_x,y,test_size=0.3,random_state=20170805 )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    for n in [ 1000,2000 ]:
        for d in [ 9,10,11 ]:
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
                    bst.save_model('/data/model/xgb_combine_{0}_{1}.model'.format( n,d ))
                    print
                    print




