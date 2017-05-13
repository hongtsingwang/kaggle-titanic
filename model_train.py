# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : model_train.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-13 13:48
# 
# =======================================================

import os
import sys
import argparse
import logging

reload(sys)
sys.setdefaultencoding('utf-8')

# parser = argparse.ArgumentParser()
# parser.add_argument()
# args = parser.parse_args()

# output = args.output
logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)   
        