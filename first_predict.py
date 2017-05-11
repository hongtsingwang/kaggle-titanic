# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : first_predict.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-11 18:10
# 
# =======================================================

import os
import sys
import logging
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre_processing
from sklearn import linear_model
from feature_engineering.basic_feature_engineering import set_Cabin_type, set_missing_ages, dummies_feature
from sklearn.ensemble import BaggingRegressor

reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


# path define
home_dir = os.getcwd()
data_dir = os.path.join(home_dir, "data")
result_dir = os.path.join(home_dir, "result")
script_dir = os.path.join(home_dir, "script")

train_file = os.path.join(data_dir, "train.csv")
test_file = os.path.join(data_dir, "test.csv")
predict_file = os.path.join(result_dir, "submission.csv")

make_dir(data_dir)
make_dir(result_dir)
make_dir(script_dir)

data_train = pd.read_csv(train_file)
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
df = dummies_feature(data_train)

scaler = pre_processing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)


# test_data处理
data_test = pd.read_csv(test_file)
data_test.loc[(data_test['Fare'].isnull()), 'Fare'] = 0
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

# 根据特征属性X预测年龄
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
df_test = dummies_feature(data_test)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(
    df_test['Fare'], fare_scale_param)


train_df = df.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

y = train_np[:, 0]
X = train_np[:, 1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8,
                               max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(
    regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(predict_file, index=False)
