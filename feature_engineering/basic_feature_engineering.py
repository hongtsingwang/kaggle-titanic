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
    return rfr


def set_missing_ages_2(df):
    """
    根据姓名中的Mr Mrs Miss 等的平均值进行填充
    """
    age_df = df[["Name","Age"]]
    salutation_list = ["Mr.","Mrs.","Miss.", "Master.", "Dr."]
    for salutation in salutation_list:
        age_predict = int(df.ix[df['Name'].apply(lambda x: salutation in x) & df['Age'].notnull(),"Age"].mean())
        df.loc[(df['Age'].isnull()) & df['Name'].apply(lambda x: salutation in x), "Age"] = age_predict
        print "%s age:%d" % (salutation,age_predict)
    df.loc[(df['Age'].isnull())] = 0 # 代表无法预测其年龄
    return df


def age_partition(df):
    """
    以年龄段分隔
    """
    bins = [0,10,20,30,40,60,80]
    labels = [1,2,3,4,5,6]
    ages = pd.cut(df.Age,bins,labels=labels)
    age_dummies = pd.get_dummies(ages, prefix="Age")
    df = pd.concat([df, age_dummies], axis=1)
    return df
    

def set_Cabin_type(df):
    """
    
    :param df: 
    :return: 
    """
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def dummies_feature(data_frame):
    data_frame = set_missing_ages_2(data_frame)
    data_frame = age_partition(data_frame)
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


def create_feature(file_name):
    data_frame = pd.read_csv(file_name, header=0)
    data_frame.loc[(data_frame['Fare'].isnull()), 'Fare'] = 0
    data_frame = set_Cabin_type(data_frame)
    data_frame = dummies_feature(data_frame)
    return data_frame

