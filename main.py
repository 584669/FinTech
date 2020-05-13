import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from datetime import datetime
from sklearn.linear_model import BayesianRidge, HuberRegressor,LinearRegression,LogisticRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def con_trd():
    trd_train = pd.read_csv('data/训练数据集_trd.csv')
    trd_test = pd.read_csv('data/评分数据集_trd_b.csv')
    data = trd_train.append(trd_test, sort=False)
    data.to_csv('data/训练数据集_trd_con.csv', index=0)

    beh_train = pd.read_csv('data/训练数据集_beh.csv')
    beh_test = pd.read_csv('data/评分数据集_beh_b.csv')
    beh_train.columns = ['id', 'flag', 'page_no', 'page_tm', 'none']
    beh_test.columns = ['id', 'page_no', 'page_tm', 'none']
    del beh_test['none']
    del beh_train['flag']
    del beh_train['none']
    data = beh_train.append(beh_test, sort=False,ignore_index=True)
    data.to_csv('data/训练数据集_beh_con.csv', index=0)
def xgb_cv(train_, test_,  y_train):
    params = {
        'eta': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': True,
        'nthread': 4,
        'n_estimators': 20000,
        'gamma': 0.1,
        'min_child_weight': 25,
        'num_threads': 8,
        'alpha': 0.18,
        'lambda': 0.23,
        'colsample_bylevel': 0.8,
    }
    train=train_.copy()
    test=test_.copy()

    X_train = train.values
    X_test = test.values
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))
    folds = KFold(n_splits=5, shuffle=True, random_state=2020)
    splits = folds.split(X_train, y_train)
    for fold_, (trn_idx, val_idx) in enumerate(splits):
        print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=100, params=params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    print("xgb auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_xgb)))
    return oof_xgb,predictions_xgb

def lgb_cv(train_, test_,y_train):
    params = {
        'num_leaves': 64,
        'min_data_in_leaf': 50,
        'objective': 'binary',  # 'binary'
        'max_depth': 6,
        'learning_rate': 0.01,
        "boosting": "gbdt",  # dart,gbdt,goss,rf
        "feature_fraction": 0.8,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "bagging_seed": 11,
        "metric": 'auc',
        "num_threads": 8,
        "verbosity": -1,
        "lambda_l1": 0.5,
        "lambda_l2": 5,
    }
    train = train_.copy()
    test = test_.copy()

    X_train = train.values
    X_test = test.values
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    folds = KFold(n_splits=5, shuffle=True, random_state=2020)
    splits = folds.split(X_train, y_train)
    for fold_, (trn_idx, val_idx) in enumerate(splits):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                        early_stopping_rounds=200)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    print("lgb auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_lgb)))
    return oof_lgb,predictions_lgb
def stack(*avg):
    train_stack = np.vstack(avg[0]).transpose()
    test_stack = np.vstack(avg[1]).transpose()
    y_train=avg[2]
    folds_stack = StratifiedKFold(n_splits=10, shuffle=True, random_state=8888)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
        print("fold :", fold_ + 1)
        trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
        val_data, val_y = train_stack[val_idx], y_train[val_idx]
        stacking = BayesianRidge()
        stacking.fit(trn_data, trn_y)
        oof_stack[val_idx] = stacking.predict(val_data)
        predictions += stacking.predict(test_stack) / folds_stack.n_splits

    print("stacking auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_stack)))
    return predictions

def transition_day(x):
    x=x.split()[0]
    x=x.split('-')
    ana=(int(x[1])-5)*30+int(x[2])
    return ana


def transition_week(x):
    return datetime.strptime(x, "%Y-%m-%d").weekday()

def transition_trd():
    data=pd.read_csv("data/训练数据集_trd_con.csv")
    data.drop_duplicates(keep='last', inplace=True)
    train = pd.DataFrame(data.groupby(['id']).size()).reset_index()
    train.columns = ['id', 'id_count']

    Dat_Flg1_Cd = pd.DataFrame(data.groupby(['id', 'Dat_Flg1_Cd']).sum()).reset_index()[
        ['id', 'Dat_Flg1_Cd', 'cny_trx_amt']]
    for i in list(set(Dat_Flg1_Cd['Dat_Flg1_Cd'])):
        name = "Dat_Flg1_Cd_{}".format(i)
        Dat = pd.DataFrame(Dat_Flg1_Cd[Dat_Flg1_Cd['Dat_Flg1_Cd'] == i])[['id','cny_trx_amt']]
        Dat.columns=['id',name]
        Dat.fillna(0)
        Dat=Dat.groupby(['id']).sum()
        train=train.merge(Dat,on=['id'], how='left')

    Trx_Cod2_Cd = pd.DataFrame(data.groupby(['id', 'Trx_Cod2_Cd']).size()).reset_index()
    Trx_Cod2_Cd.columns = ['id', 'Trx_Cod2_Cd', 'Trx_Cod2_Cd_size']
    for i in list(set(Trx_Cod2_Cd['Trx_Cod2_Cd'])):
        name = "Trx_Cod2_Cd_count_{}".format(i)
        Trx = pd.DataFrame(Trx_Cod2_Cd[Trx_Cod2_Cd['Trx_Cod2_Cd'] == i])[['id','Trx_Cod2_Cd_size']]
        Trx.columns=['id',name]
        Trx.fillna(0)
        Trx = Trx.groupby(['id']).sum()
        train =train.merge(Trx, on=['id'], how='left')

    Trx_Cod1_Cd = pd.DataFrame(data.groupby(['id', 'Trx_Cod1_Cd']).size()).reset_index()
    Trx_Cod1_Cd.columns = ['id', 'Trx_Cod1_Cd', 'Trx_Cod1_Cd_count']
    for i in list(set(Trx_Cod1_Cd['Trx_Cod1_Cd'])):
        name = "Trx_Cod1_Cd_count_{}".format(i)
        Trx = pd.DataFrame(Trx_Cod1_Cd[Trx_Cod1_Cd['Trx_Cod1_Cd'] == i])[['id', 'Trx_Cod1_Cd_count']]
        Trx.columns = ['id', name]
        Trx.fillna(0)
        Trx = Trx.groupby(['id']).sum()
        train = train.merge(Trx, on=['id'], how='left')

    Dat_Flg3_Cd = pd.DataFrame(data.groupby(['id', 'Dat_Flg3_Cd']).size()).reset_index()
    Dat_Flg3_Cd.columns = ['id', 'Dat_Flg3_Cd', 'Dat_Flg3_Cd_count']
    for i in list(set(Dat_Flg3_Cd['Dat_Flg3_Cd'])):
        name = "Dat_Flg3_Cd_count_{}".format(i)
        Dat = pd.DataFrame(Dat_Flg3_Cd[Dat_Flg3_Cd['Dat_Flg3_Cd'] == i])[['id', 'Dat_Flg3_Cd_count']]
        Dat.columns = ['id', name]
        Dat.fillna(0)
        Dat = Dat.groupby(['id']).sum()
        train = train.merge(Dat, on=['id'], how='left')

    Dat_Flg1_Cd_and_Dat_Flg3_Cd = pd.DataFrame(data.groupby(['id', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd']).sum()).reset_index()
    Dat_Flg1_Cd_and_Dat_Flg3_Cd = Dat_Flg1_Cd_and_Dat_Flg3_Cd[['id', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'cny_trx_amt']]
    for i in list(set(Dat_Flg1_Cd_and_Dat_Flg3_Cd['Dat_Flg1_Cd'])):
        for j in list(set(Dat_Flg1_Cd_and_Dat_Flg3_Cd['Dat_Flg3_Cd'])):
            name = "Dat_Flg1_Cd_and_Dat_Flg3_Cd_{}_{}".format(i, j)
            Dat = pd.DataFrame(Dat_Flg1_Cd_and_Dat_Flg3_Cd[Dat_Flg1_Cd_and_Dat_Flg3_Cd['Dat_Flg1_Cd'] == i])
            Dat = pd.DataFrame(Dat[Dat['Dat_Flg3_Cd'] == j])
            Dat= Dat[['id', 'cny_trx_amt']]
            Dat['cny_trx_amt'] = Dat['cny_trx_amt']
            Dat.columns = ['id', name]
            train = train.merge(Dat, on=['id'], how='left')

    Dat_Flg1_Cd_and_Trx_Cod1_Cd= pd.DataFrame(data.groupby(['id', 'Dat_Flg1_Cd', 'Trx_Cod1_Cd']).sum()).reset_index()
    Dat_Flg1_Cd_and_Trx_Cod1_Cd =Dat_Flg1_Cd_and_Trx_Cod1_Cd[['id', 'Dat_Flg1_Cd', 'Trx_Cod1_Cd', 'cny_trx_amt']]
    for i in list(set(Dat_Flg1_Cd_and_Trx_Cod1_Cd['Dat_Flg1_Cd'])):
        for j in list(set(Dat_Flg1_Cd_and_Trx_Cod1_Cd['Trx_Cod1_Cd'])):
            name = "log_flg1_cod1_{}_{}".format(i, j)
            Dat = pd.DataFrame(Dat_Flg1_Cd_and_Trx_Cod1_Cd[Dat_Flg1_Cd_and_Trx_Cod1_Cd['Dat_Flg1_Cd'] == i])
            Dat = pd.DataFrame(Dat[Dat['Trx_Cod1_Cd'] == j])
            Dat = Dat[['id', 'cny_trx_amt']]
            Dat['cny_trx_amt'] = Dat['cny_trx_amt']
            Dat.columns = ['id', name]
            train = train.merge(Dat, on=['id'], how='left')
    data['trx_tm_day'] = data['trx_tm'].apply(lambda x: transition_day(x))
    start= pd.DataFrame(data.groupby(['id']).max()).reset_index()
    end = pd.DataFrame(data.groupby(['id']).min()).reset_index()
    time=start[['id','trx_tm_day']]
    time['trx_tm_day_end']=end['trx_tm_day']
    time.columns = ['id', 'trx_tm_day_start', 'trx_tm_day_end']
    time['start_end'] =time['trx_tm_day_end']-time['trx_tm_day_start']
    train = train.merge(time, on=['id'], how='left')
    print(train.head(5))
    return train

def transition_tag(path,data_trd):
    data_tag = pd.read_csv(path)
    data_tag.replace("\\N", np.nan, inplace=True)
    data_tag.replace("\\\\N", np.nan, inplace=True)
    del data_tag['edu_deg_cd']
    del data_tag['deg_cd']
    data_tag = data_tag.merge(data_trd, on=['id'], how='left')


    data_tag['cur_debit_min_opn_dt_cnt'] = data_tag['cur_debit_min_opn_dt_cnt'] / 30
    data_tag['cur_credit_min_opn_dt_cnt'] = data_tag['cur_credit_min_opn_dt_cnt'] / 30
    mapping = {'M': 0, 'F': 1}
    data_tag['gdr_cd'] = data_tag['gdr_cd'].map(mapping)
    mapping = {'O': 0, 'A': 1, 'Z': 2, 'B': 3, '~': 4}
    data_tag['mrg_situ_cd'] = data_tag['mrg_situ_cd'].map(mapping)
    mapping = {'C': 0, 'Z': 1, '31': 2, 'G': 3, '30': 4, 'F': 5, 'D': 6}
    data_tag['acdm_deg_cd'] = data_tag['acdm_deg_cd'].map(mapping)
    return data_tag
def deal_result(x):
    if(x>=0.8):
        x=1
    elif(x<=0):
        x=0
    return x
con_trd()
data_trd = transition_trd()
train=transition_tag("data/训练数据集_tag.csv",data_trd)
test=transition_tag("data/评分数据集_tag_b.csv",data_trd)
result= pd.DataFrame()
result['score']= test['id']
del train['id']
del test['id']
y_train = train['flag']
del train['flag']
train_stack = []
test_stack = []
oof_xgb, predictions_xgb = xgb_cv(train, test,y_train)
train_stack.append(oof_xgb)
test_stack.append(predictions_xgb)
oof_lgb, predictions_lgb = lgb_cv(train, test,y_train)
train_stack.append(oof_lgb)
test_stack.append(predictions_lgb)
result['score']= stack(train_stack, test_stack, y_train)
result['score']= result['score'].apply(lambda x: deal_result(x))
with open('ana.txt', 'w', encoding='utf-8') as f:
    for i in range(len(result)):
        f.write(str(result.loc[i, 'id']) + '\t' + str(result.loc[i, 'score']) + '\n')