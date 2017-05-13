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
from sklearn import linear_model
from feature_engineering.basic_feature_engineering import create_feature
from sklearn.ensemble import BaggingRegressor

reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)

# path define
home_dir = os.getcwd()
data_dir = os.path.join(home_dir, "data")
result_dir = os.path.join(home_dir, "result")
script_dir = os.path.join(home_dir, "script")

train_file = os.path.join(data_dir, "train.csv")
train = pd.read_csv(train_file, header=0)
test_file = os.path.join(data_dir, "test.csv")
test = pd.read_csv(test_file, header=0)
predict_file = os.path.join(result_dir, "submission.csv")

# feature engieering之前的必要的数据提取
pre_statistics = dict()
pre_statistics['fare_median'] = train['Fare'].median()

# 特征工程部分, 取文件，提取feature，做预处理
data_train = create_feature(train_file,pre_statistics)
data_test = create_feature(test_file,pre_statistics)
PassengerId = test['PassengerId']

# 模型训练部分, 训练模型
train_df = data_train.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# 模型预测部分， 预测结果
y = train_np[:, 0]
X = train_np[:, 1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8,
                               max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = data_test.filter(
    regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)


# Generate Submission File
StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,
                                   'Survived': predictions})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)


# 结果提交部分， 提交结果
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
), 'Survived': predictions.astype(np.int32)})
result.to_csv(predict_file, index=False)
