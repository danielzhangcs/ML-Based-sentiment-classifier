#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:55:14 2018

@author: Daniel
"""

from ml_sentiment import *
import eli5
import pandas as pd
filedir = 'data'
train_data, dev_data = get_training_and_dev_data(filedir)
pd.set_option('display.max_colwidth', -1)

def get_error_type(pred, label):
    # return the type of error: tp,fp,tn,fn
    if pred == label:
        return "tp" if pred == '1' else "tn"
    return "fp" if pred == '1' else "fn"


# Change this for your different classifiers
classifier1 = load_classifier('lr_default.pkl')
#classifier2 = load_classifier('rf_custom.pkl')