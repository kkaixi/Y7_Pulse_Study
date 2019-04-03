# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:07:32 2019

@author: tangk
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from PMG.read_data import initialize
from PMG.COM.plotfuns import *
from PMG.COM.get_props import *
from PMG.COM.arrange import *

directory = 'P:\\Data Analysis\\Projects\\Y7 Pulse Study\\'
cutoff = range(100, 1600)

channels = ['12HEAD0000Y7ACXA',
            '12HEAD0000Y7ACYA',
            '12HEAD0000Y7ACZA',
            '12HEAD0000Y7ACRA',
            '12CHST0000Y7ACXC',
            '12CHST0000Y7ACYC',
            '12CHST0000Y7ACZC',
            '12CHST0000Y7ACRC',
            '12CHST0000Y7DSXB',
            '12PELV0000Y7ACXA',
            '12PELV0000Y7ACYA',
            '12PELV0000Y7ACZA',
            '12PELV0000Y7ACRA',
            '12SEBE0000B3FO0D',
            '12SEBE0000B6FO0D',
            'S0SLED000000ACXD',
            'S0SLED000000VEXD']

drop = ['SE18-0056',
        'SE18-0056_2',
        'SE18-0056_3',
        'SE17-0029_3',
        'SE17-0029_4']

table, t, chdata = initialize(directory,channels, cutoff, drop=drop)

#%% feature extraction
def get_all_features(write_csv=False):
    i_to_t = get_i_to_t(t)
    feature_funs = {'Min_': [get_min],
                    'Max_': [get_max]} 
    features = pd.concat(chdata.chdata.get_features(feature_funs).values(),axis=1,sort=True)

    if write_csv:
        features.to_csv(directory + 'features.csv')
    return features

features = get_all_features(write_csv=False)

#%% 
import seaborn as sns
sns.barplot(x='Model', y='Chest_3ms', hue='Pulse', data=table)