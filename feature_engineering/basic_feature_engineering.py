# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : basic_feature_engineering.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-11 18:12
# 
# =======================================================

import re

import os
import sys
import argparse
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

reload(sys)
sys.setdefaultencoding('utf-8')

"""
一些基本的特征工程工作
"""


def set_missing_ages(df):
    """
    利用随机森林的算法填充缺失的年龄
    :param df: 
    :return: 
    """
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    return rfr


def set_missing_ages_2(df):
    """
    根据姓名中的Mr Mrs Miss 等的平均值进行填充
    """
    age_df = df[["Name", "Age"]]
    salutation_list = ["Mr.", "Mrs.", "Miss.", "Master.", "Dr."]
    for salutation in salutation_list:
        age_predict = int(df.ix[df['Name'].apply(lambda x: salutation in x) & df['Age'].notnull(), "Age"].mean())
        df.loc[(df['Age'].isnull()) & df['Name'].apply(lambda x: salutation in x), "Age"] = age_predict
        print "%s age:%d" % (salutation, age_predict)
    df.loc[(df['Age'].isnull())] = 0  # 代表无法预测其年龄
    return df


def age_partition(df):
    """
    以年龄段分隔
    """
    bins = [0, 16, 32, 48, 64,80]
    labels = [1, 2, 3, 4, 5]
    ages = pd.cut(df.Age, bins, labels=labels)
    age_dummies = pd.get_dummies(ages, prefix="Age")
    df = pd.concat([df, age_dummies], axis=1)
    return df


def cabin_feature(data_frame):
    """
    
    :param df: 
    :return: 
    """
    data_frame.loc[(data_frame.Cabin.notnull()), 'Cabin'] = "Yes"
    data_frame.loc[(data_frame.Cabin.isnull()), 'Cabin'] = "No"
    dummies_cabin = pd.get_dummies(data_frame['Cabin'], prefix='Cabin')
    return dummies_cabin


def pclass_feature(data_frame):
    dummies_Pclass = pd.get_dummies(data_frame['Pclass'], prefix='Pclass')
    return dummies_Pclass


def sex_feature(data_frame):
    """
    
    :param data_frame: 
    :return: 
    """
    dummies_Sex = pd.get_dummies(data_frame['Sex'], prefix='Sex')
    return dummies_Sex


def embark_feature(data_frame):
    """
    
    :param data_frame: 
    :return: 
    """
    data_frame['Embarked'].fillna('S')
    dummies_Embarked = pd.get_dummies(
        data_frame['Embarked'], prefix='Embarked')
    return dummies_Embarked


def age_feature(data_frame):
    data_frame.loc[(data_frame['Fare'].isnull()), 'Fare'] = 0
    data_frame = age_partition(data_frame)
    return data_frame


def fare_feature(data_frame, fare_median):
    """
    票价相关feature
    :param data_frame: 
    :return: 
    """
    data_frame.loc[(data_frame['Fare'].isnull()), 'Fare'] = fare_median
    bins = [0, 7.91, 14.454, 31, 10000000000]
    data_frame["fare_cut"] = pd.cut(data_frame["Fare"], bins=bins, labels=[0, 1, 2, 3, 4])
    fare_frame = data_frame["Fare"]
    return fare_frame


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def name_feature(data_frame):
    """
    根据乘客的名称提取的feature
    :param data_frame: 
    :return: 
    """
    data_frame["name_length"] = data_frame["Name"].apply(len)
    data_frame['Title'] = data_frame['Name'].apply(get_title)
    data_frame['Title'] = data_frame['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_frame['Title'] = data_frame['Title'].replace('Mlle', 'Miss')  # Mlle 和 Miss是同义的
    data_frame['Title'] = data_frame['Title'].replace('Ms', 'Miss')
    data_frame['Title'] = data_frame['Title'].replace('Mme', 'Mrs')
    return data_frame[["name_length", "Title"]]


def family_feature(data_frame):
    """
    根据乘客的子女和亲属情况确定feature
    :param data_frame: 
    :return: 
    """
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch'] + 1
    data_frame["IsAlone"] = 0
    data_frame.loc[data_frame['FamilySize'] == 1, 'IsAlone'] = 1
    return data_frame[['FamilySize', 'IsAlone']]


def create_feature(file_name, pre_statistics):
    """
    用于创建train和test相对独立， 不发生依赖的feature
    :param file_name: 
    :return: 
    """
    data_frame = pd.read_csv(file_name, header=0)
    fare_median = pre_statistics["fare_median"]
    fare_frame = fare_feature(data_frame, fare_median)
    cabin_frame = cabin_feature(data_frame)
    pclass_frame = pclass_feature(data_frame)
    sex_frame = sex_feature(data_frame)
    df = pd.concat([data_frame, fare_frame, cabin_frame,
                    sex_frame, pclass_frame], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket',
             'Cabin', 'Embarked','SibSp'], axis=1, inplace=True)
    return data_frame
