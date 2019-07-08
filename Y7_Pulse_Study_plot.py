# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:34:53 2019

@author: tangk
"""
import numpy as np
import pandas as pd
from initialize import dataset
from PMG.COM.arrange import arrange_by_group
from PMG.COM.plotfuns import *
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly_express as px
from PMG.COM.linear_model import *
from sklearn.linear_model import *

dataset.get_data(['timeseries','features'])
#%%
#plot_channels = dataset.channels
#grouped = dataset.table.groupby('Model')
#for ch in plot_channels:
#    fig, axs = get_axes(3)
#    for grp, ax in zip(grouped, axs.flatten()):
#        x = arrange_by_group(grp[1], dataset.timeseries[ch], 'Pulse')
#        ax = plot_overlay(ax, dataset.t, x)
#        ax = set_labels(ax, {'title': grp[0], 'legend': {}})
#    fig.suptitle(ch)
#    plt.show()
#    plt.close(fig)

#%% find predictors
x = dataset.features.select_dtypes(np.float64)
y = dataset.features['Min_12CHST0000Y7ACXC'].to_frame()
x = drop_correlated(x,y)
x = pd.concat((x, dataset.table.select_dtypes(np.float64).drop(['Head_3ms','Chest_3ms'], axis=1)), axis=1)

x, y = preprocess_data(x, y)
models = iter([Ridge(max_iter=i) for i in range(1, 30)])

vs = VariableSelector(x, y, models)
vs.find_variables()
