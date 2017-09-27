# coding=utf-8
import gc
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def trans_data( tr_x,isTrain=True ):
    print('开始构造训练集---------------------------------------------------')
    path = '/data/cache/xgb.pkl'
    if not isTrain:
        path = '/data/cache/xgb_te.pkl'
    if os.path.exists( path ):
        return pd.read_pickle( path )
    else:
        ######################################################### fill nan #################################################################################
        def do_province():
            province = ['黑龙江','吉林省','辽宁省','河北省','河南省','山东省','江苏省','山西省','陕西省','甘肃省','四川省'
                        ,'青海省','湖南省','湖北省','江西省','安徽省','浙江省','福建省','广东省','广西省','贵州省','云南省','海南省'
                        ,'内蒙古','新疆','宁夏','西藏','北京市','天津市','重庆市' ] #
            p_dict = pd.DataFrame( province,columns=['name'] )#去掉上海和港澳台
            p_dict['index'] = p_dict.index
            p_dict.index = p_dict['name']
            p_dict = p_dict['index'].to_dict()
            p_array = np.zeros((len(tr_x), 30))
            i = 0
            for x in tr_x['漫出省份'].values:
                for p in x.split(','):
                    if p in p_dict:
                        p_array[i][p_dict[p]] = 1
                i += 1
            return pd.DataFrame( p_array,columns=['to_province_{0}'.format(province[i]) for i in range( 30 ) ] )
        p_df = do_province()
        tr_x = pd.concat( [tr_x,p_df],axis=1,join='inner' )
        tr_x['to_area_华北'] = tr_x['to_province_北京市'] + tr_x['to_province_天津市'] + tr_x['to_province_河北省'] + tr_x['to_province_山西省']+tr_x['to_province_内蒙古']
        tr_x['to_area_东北'] = tr_x['to_province_辽宁省'] + tr_x['to_province_吉林省'] + tr_x['to_province_黑龙江']
        tr_x['to_area_华东'] = tr_x['to_province_江苏省'] + tr_x['to_province_浙江省'] + tr_x['to_province_安徽省'] +tr_x['to_province_福建省'] +tr_x['to_province_江西省']+tr_x['to_province_山东省']
        tr_x['to_area_华中'] = tr_x['to_province_河南省'] + tr_x['to_province_湖北省'] + tr_x['to_province_湖南省']
        tr_x['to_area_华南'] = tr_x['to_province_广东省'] + tr_x['to_province_广西省'] + tr_x['to_province_海南省' ]
        tr_x['to_area_西北'] = tr_x['to_province_陕西省'] + tr_x['to_province_甘肃省'] + tr_x['to_province_青海省'] +tr_x['to_province_宁夏'] +tr_x['to_province_新疆' ]
        tr_x['to_area_西南'] = tr_x['to_province_重庆市'] + tr_x['to_province_四川省'] + tr_x['to_province_贵州省'] +tr_x['to_province_云南省'] +tr_x['to_province_西藏' ]
        tr_x['用户更换手机频次'] = tr_x['用户更换手机频次'].fillna(0)
        tr_x['手机品牌'] = tr_x['手机品牌'].fillna('未知')
        ######################################################### handle category feature ########################################################
        def do_brand(x):
            if x in ['苹果', '华为', '小米', '三星', '欧珀', '荣耀', '维沃', '魅族', 'CIB', '乐视']:
                return x
            return '其他'
        tr_x['手机品牌'] = tr_x['手机品牌'].apply(do_brand)
        mb = pd.get_dummies(tr_x['手机品牌'], prefix='mobile_brand')
        tr_x['是否有跨省行为'] = tr_x['是否有跨省行为'].apply(lambda s: 1 if s == '是' else 0)
        tr_x['是否有出境行为'] = tr_x['是否有出境行为'].apply(lambda s: 1 if s == '是' else 0)
        del tr_x['手机品牌']
        del tr_x['性别']
        del tr_x['手机终端型号']
        del tr_x['漫入省份']
        ###LR中拟合出来权重低的在e-6量级
        del tr_x['必应搜索']
        del tr_x['SuningEbuy']
        del tr_x['乐安全']
        del tr_x['爱奇艺动画屋']
        del tr_x['百度视频']
        del tr_x['Nokia Browser']
        del tr_x['点心壁纸']
        del tr_x['安智市场']
        del tr_x['QQ输入法']
        del tr_x['美图秀秀']
        del tr_x['Adobe AIR']
        del tr_x['直播吧']
        del tr_x['赢家理财高端版']
        del tr_x['易到用车']
        del tr_x['拼立得']
        del tr_x['搜房网房天下']
        del tr_x['支付宝']
        del tr_x['赶集网']
        del tr_x['aas limitess success magazine']
        del tr_x['新征途']
        del tr_x['天涯社区']
        del tr_x['内涵段子']
        del p_df
        gc.collect()
        tr_x = pd.concat([tr_x, mb], axis=1, join='inner')  # axis=1 是行
        ######################################################### new feature dig ########################################################
        tr_x['num_province'] = tr_x['漫出省份'].apply(lambda x: len(str(x).split(',')))
        tr_x['nnz'] = 0
        for col in list(tr_x.columns):
            tr_x['nnz'] += tr_x[col].apply(lambda x: 1 if x != 0 else 0)
        del tr_x['漫出省份']
        gc.collect()
        pd.to_pickle( tr_x,path )
        return tr_x

def sub_xgb( n,d ):
    tr_x = pd.read_csv('/data/q1/data_train.csv')
    tr_x = trans_data(tr_x)
    del tr_x['用户标识']
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
    clf.save_model('/data/model/xgb_{0}_{1}.model'.format(n, d))

    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model('/data/model/xgb_v2_0.1_1000.model')
    sub = pd.DataFrame()
    te_x = pd.read_csv('/data/q1/data_test.csv')
    sub['IMEI'] = te_x['用户标识']
    te_X = trans_data(te_x,False)
    del te_X['用户标识']
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
    s.to_csv('/data/cache/feature_scores.csv',index=False)
    te_X = xgb.DMatrix(te_X)
    sub['SCORE'] = clf.predict(te_X)
    sub.to_csv('/data/res/result_xgb_{0}_{1}.csv'.format( n,d ), index=False)

sub_xgb(1000,11)