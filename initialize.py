# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:07:32 2019

@author: tangk
"""
from PMG.read_data import PMGDataset
from PMG.COM.get_props import get_peaks
import json
import pandas as pd

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

table_filters = {'drop': ['SE18-0056',
                          'SE18-0056_2',
                          'SE18-0056_3',
                          'SE17-0029_3',
                          'SE17-0029_4']}
preprocessing = None

# initialize the dataset and specify channels, filters, and preprocessing
dataset = PMGDataset(directory, channels=channels, cutoff=cutoff, verbose=False)
dataset.table_filters = table_filters
dataset.preprocessing = preprocessing


if __name__=='__main__': 
    # if running the script, get the data
    dataset.get_data(['timeseries'])
    table = dataset.table
    features = get_peaks(dataset.timeseries)
    features = pd.concat((features, table), axis=1)
    features.to_csv(directory + 'features.csv')

#    table, t, chdata = initialize(directory,channels, cutoff, drop=drop)
    #%% json file specifying statistical tests to be done
    to_JSON = {'project_name': 'Y7_Pulse_Study',
               'directory'   : directory,
               'cat'     : {'all_tests': table.index.tolist()},
               'data'    : [{'features'   : 'features.csv'}],
               'test'    : [{'name': 'LME_all',
                             'test1_name': 'all_tests',
                             'variables': ['Pulse','Model'],
                             'formula': 'Pulse + (1|Model)',
                             'null_formula': '(1|Model)',
                             'testname': 'lmer',
                             'data': 'features',
                             'model_args': None,
                             'test_args': None}],
                'test2'  : None}    
    
    for test in to_JSON['test']:
        test['test1'] = to_JSON['cat'][test['test1_name']]
        if 'test2_name' in test:
            test['test2'] = to_JSON['cat'][test['test2_name']]
    with open(directory+'params.json','w') as json_file:
        json.dump(to_JSON,json_file)

