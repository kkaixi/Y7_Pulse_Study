# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:34:53 2019

@author: tangk
"""

from initialize import dataset
from PMG.COM.arrange import arrange_by_group
from PMG.COM.plotfuns import *
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly_express as px

dataset.get_data(['timeseries','features'])
#%%
plot_channels = dataset.channels
grouped = dataset.table.groupby('Model')
for ch in plot_channels:
    fig, axs = get_axes(3)
    for grp, ax in zip(grouped, axs.flatten()):
        x = arrange_by_group(grp[1], dataset.timeseries[ch], 'Pulse')
        ax = plot_overlay(ax, dataset.t, x)
        ax = set_labels(ax, {'title': grp[0], 'legend': {}})
    fig.suptitle(ch)
    plt.show()
    plt.close(fig)
