# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : basic_feature_engineering.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-11 18:12
# 
# =======================================================

import os
import sys
import argparse
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

reload(sys)
sys.setdefaultencoding('utf-8')


def set_missing_ages(df):
    """
    利用随机森林的算法填充缺失的年龄
    :param df: 
    :return: 
    """
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    predicted_age = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predicted_age
    return df, rfr


def set_Cabin_type(df):
    """
    
    :param df: 
    :return: 
    """
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    return df


def dummies_feature(data_frame):
    data_frame = set_Cabin_type(data_frame)
    dummies_Cabin = pd.get_dummies(data_frame['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(
        data_frame['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_frame['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_frame['Pclass'], prefix='Pclass')
    df = pd.concat([data_frame, dummies_Cabin, dummies_Embarked,
                    dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket',
             'Cabin', 'Embarked'], axis=1, inplace=True)
    return df
