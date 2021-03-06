# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : xgboost_training.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-13 15:15
# 
# =======================================================

import os
import sys
import argparse
import logging

sys.path.append("../")
import numpy as np
from sklearn.cross_validation import KFold
from SklearnHelper import SklearnHelper
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb

reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)


def get_oof(clf, x_train, y_train, x_test, ntrain, ntest):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def training(train, test):
    """
    
    :param train: 
    :return: 
    """
    # Some useful parameters which will come in handy later on
    ntrain = train.shape[0]
    ntest = test.shape[0]
    SEED = 0  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

    # Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
    y_train = train['Survived'].ravel()
    train = train.drop(['Survived'], axis=1)
    x_train = train.values  # Creates an array of the train data
    x_test = test.values  # Creats an array of the test data

    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }

    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 500,
        # 'max_features': 0.2,
        'max_depth': 5,
        sh whq@10.11.147.21
        min_samples_leaf': 2,
        'verbose': 0
    }

    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'linear',
        'C': 0.025
    }

    # Create 5 objects that represent our 4 models
    rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

    # Create our OOF train and test predictions. These base results will be used as new features
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
    ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
    gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
    svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

    x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
    x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

    gbm = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=4,
        min_child_weight=2,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1).fit(x_train, y_train)

    return gbm
