# coding=utf-8
import os
from sklearn import metrics
import pandas as pd
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
    mc = pd.get_dummies(tr_x['手机品牌'] + '_' + tr_x['大致消费水平'].astype(str), prefix='mobile_cross_arpu')
    tr_x = pd.concat([tr_x, ma,mc], axis=1, join='inner')
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

def rf_eval(  ):
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    del tr_x['用户标识']
    tr_x = trans_data( tr_x )
    X = tr_x.values
    label = pd.read_csv('/data/q1/label.csv')
    y = label['是否去过迪士尼'].values
    X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.3,random_state=20170730 )
    best_param = []
    best_score = 0.0
    for n in [ 300,500,800 ]:
        for d in [ 6,7,8 ]:
            rf_clf = RandomForestClassifier(n_estimators=n, max_depth=d,min_samples_leaf = 5, random_state = 20170730)
            rf_clf.fit( X_train,y_train )
            pd.to_pickle( rf_clf,'/data/cache/rf_{}_{}.pkl'.format( n,d ) )
            pred = rf_clf.predict_proba(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred[:,1], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print 'n_estimators={},max_depth={},AUC={}'.format(n, d, auc)
            if best_score< auc:
                best_score = auc
                best_param.insert(0,[n,d])
    print auc
    print( best_param )
def sub_rf():
    clf = pd.read_pickle('/data/cache/rf_800_8.pkl')
    sub = pd.DataFrame()
    te_x = pd.read_csv('/data/q1/data_test.csv')
    sub['IMEI'] = te_x['用户标识']
    del te_x['用户标识']
    te_X = trans_data(te_x)
    X = te_X.values
    rf_info = pd.DataFrame()
    rf_info['feature'] = te_X.columns
    rf_info['importance'] = clf.feature_importances_
    rf_info = rf_info.sort_values( 'importance',ascending=False )
    rf_info.to_pickle('/data/cache/rf_info.pkl')
    sub['SCORE'] = clf.predict_proba(X)[:, 1]
    sub.to_csv('/data/res/result.csv', index=False)


