# coding=utf-8
import gc
import os
from time import sleep
import numpy as np
import pandas as pd

exit_code = os.system('sh /data/proj/bin/run_xgb_stack.sh 0 >> run.log')
for i in range(1,5):
    if exit_code == 0:
        exit_code = os.system('sh /data/proj/bin/run_xgb_stack.sh {0} >> run.log'.format(i))
    else:os._exit(1)
    sleep(2)
ntrain = 844984
ntest = 211245
X_train = pd.read_csv('/data/q1/data_train.csv')
X_test = pd.read_csv('/data/q1/data_test.csv')
oof_test = np.zeros((ntest))
oof_test_skf = np.zeros((5, ntest))
t1 = pd.read_pickle('/data/cache/xgb_stack_oof_train0.pkl')
t2 = pd.read_pickle('/data/cache/xgb_stack_oof_train1.pkl')
t3 = pd.read_pickle('/data/cache/xgb_stack_oof_train2.pkl')
t4 = pd.read_pickle('/data/cache/xgb_stack_oof_train3.pkl')
t5 = pd.read_pickle('/data/cache/xgb_stack_oof_train4.pkl')
oof_train = np.vstack((t1, t2, t3,t4,t5))
train_df = pd.DataFrame(oof_train, columns=['score'])
train_df['IMEI'] = X_train['用户标识']
del X_train
gc.collect()
train_df.to_csv('/data/cache/xgb_stack_train.csv')
sub = pd.DataFrame()
sub['IMEI'] = X_test['用户标识']
del X_test
gc.collect()
for i in range(5):
    te_path = '/data/cache/xgb_stack_oof_test{0}.pkl'.format(i)
    oof_test_skf[i, :] = pd.read_pickle(te_path).reshape(1, -1)
oof_test[:] = oof_test_skf.mean(axis=0)
sub['SCORE'] = oof_test.reshape(-1, 1)
sub.to_csv('/data/res/result_xgb_stacking_v2.csv', index=False)