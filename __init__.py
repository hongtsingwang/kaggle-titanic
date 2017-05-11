# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : __init__.py.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-11 14:07
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
        