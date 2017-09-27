# coding=utf-8
import gc
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def trans_data( tr_x ):
    tr_x['用户更换手机频次'] = tr_x['用户更换手机频次'].fillna(0)
    tr_x['手机品牌'] = tr_x['手机品牌'].fillna('未知')
    def do_brand( x ):
        if x in ['苹果','华为','小米','三星','欧珀','荣耀','维沃','魅族','CIB','乐视']:
            return x
        return '其他'
    tr_x['手机品牌'] = tr_x['手机品牌'].apply(do_brand)
    mb = pd.get_dummies(tr_x['手机品牌'], prefix='mobile_brand')
    sex = pd.get_dummies(tr_x['性别'], prefix='sex')
    tr_x['是否有跨省行为'] = tr_x['是否有跨省行为'].apply( lambda s: 1 if s=='是' else 0 )
    tr_x['是否有出境行为'] = tr_x['是否有出境行为'].apply( lambda s: 1 if s=='是' else 0 )
    tr_x = pd.concat([tr_x,mb,sex],axis=1,join='inner') #axis=1 是行
    ###衍生出来的
    tr_x['nnz'] = 0
    for col in list(tr_x.columns):
        tr_x['nnz'] += tr_x[col].apply( lambda x:1 if x!=0 else 0)
    ma = pd.get_dummies( tr_x['手机品牌'] +'_'+tr_x['年龄段'].astype(str),prefix='mobile_cross_age' )
    tr_x = pd.concat([tr_x, ma], axis=1, join='inner')
    del tr_x['手机品牌']
    del tr_x['性别']
    del tr_x['手机终端型号']
    del tr_x['漫入省份']
    del tr_x['漫出省份']
    ###LR中拟合出来权重低的在e-6量级
    del tr_x['必应搜索']
    del tr_x['SuningEbuy']
    del tr_x['乐安全']
    del tr_x['爱奇艺动画屋']
    gc.collect()
    return tr_x


def lr_train(  ):#注：这道题LR调参基本没有提升
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    del tr_x['用户标识']
    tr_x = trans_data( tr_x )
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼'].values
    scaler_tr = pd.DataFrame()
    for col in list(tr_x.columns):#fitting all columns occur OOM error
        print col
        scaler_tr[col] = StandardScaler().fit_transform(tr_x[col].values.reshape(-1,1))[:,0]
    X = scaler_tr.values
    # Tolerance for stopping criteria. && for large dataset,solver should be “lbfgs”, “sag” or “newton-cg”
    clf_l1_LR = LogisticRegression(C=100, penalty='l2', tol=1e-6,max_iter=10000,solver='lbfgs')
    clf_l1_LR.fit(X,y)
    pd.to_pickle( clf_l1_LR,'/data/model/lr_v2.pkl' )
    print 'fit done'
    lr_info = pd.DataFrame()
    lr_info['col'] = tr_x.columns
    lr_info['coef'] = clf_l1_LR.coef_[0, :]
    lr_info['coef_abs'] = lr_info['coef'].abs()
    lr_info = lr_info.sort_values('coef_abs',ascending=False)
    pd.to_pickle(lr_info,'/data/cache/lr_info.pkl')


def lr_sub(  ):
    lr_train()
    clf_l1_LR = pd.read_pickle('/data/model/lr_v2.pkl')
    sub =  pd.DataFrame()
    te_x = pd.read_csv('/data/q1/data_test.csv')
    sub['IMEI'] = te_x['用户标识']
    del te_x['用户标识']
    te_X = trans_data( te_x )
    scaler_te = pd.DataFrame()
    for col in list(te_X.columns):
        scaler_te[col] = StandardScaler().fit_transform(te_X[col].values.reshape(-1,1))[:,0]
    X = scaler_te.values
    sub['SCORE'] = clf_l1_LR.predict_proba( X )[:,1]
    sub.to_csv('/data/res/result.csv',index=False)
    token = '**********'
    os.system('kesci_submit -token %s -file /data/res/result.csv' % token)
lr_sub()
