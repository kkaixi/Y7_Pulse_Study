# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:21:49 2019

For identification of predictors of chest response

@author: tangk
"""
from PMG.read_data import PMGDataset
from PMG.COM.get_props import get_peaks, smooth_data
import json
import pandas as pd
from string import ascii_lowercase
import statsmodels.api as sm
import seaborn as sns

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

table_filters = {'query': 'Type==\'LB\' and Pulse==\'Regular\''}
preprocessing = None


dataset = PMGDataset(directory, channels=channels, cutoff=cutoff, verbose=False)
dataset.table_filters = table_filters
dataset.preprocessing = preprocessing

#smoothing
#dataset.get_data(['timeseries'])
#dataset.timeseries[['12PELV0000Y7ACXA_sm','12PELV0000Y7ACYA_sm','12PELV0000Y7ACZA_sm']] = dataset.timeseries[['12PELV0000Y7ACXA','12PELV0000Y7ACYA','12PELV0000Y7ACZA']].applymap(smooth_data)
#
#features = get_peaks(dataset.timeseries)
#dataset.features = features

dataset.get_data([])
dataset.features = pd.DataFrame()
faro = pd.read_excel(directory + 'Faro_Measurements.xlsx').set_index('SE').loc[dataset.table.index]

# get points
faro[faro.filter(like='_x').columns] = faro.filter(like='_x').applymap(lambda x: -x)
faro['neck_pubis_dx'] = faro['Pubis_x'] - faro['Neck_x']
faro['neck_pubis_dy'] = faro['Pubis_y'] - faro['Neck_y']
faro['neck_pubis_dz'] = faro['Pubis_z'] - faro['Neck_z']
faro['neck_pubis_angle_Y'] = faro['neck_pubis_dx']/faro['neck_pubis_dz']
faro['sb_neck_dx'] = faro['Neck_x'] - faro['SB_inboard_x']
faro['sb_neck_dy'] = faro['Neck_y'] - faro['SB_inboard_y']
faro['sb_dx'] = faro['SB_outboard_x'] - faro['SB_inboard_x']
faro['sb_dy'] = faro['SB_outboard_y'] - faro['SB_inboard_y']
faro['knee_pubis_dx'] = faro['Knee_x'] - faro['Pubis_x']
faro['knee_pubis_dy'] = faro['Knee_y'] - faro['Pubis_y']
faro['knee_pubis_dz'] = faro['Knee_z'] - faro['Pubis_z']
faro['thigh_angle_Y'] = faro['knee_pubis_dz']/faro['knee_pubis_dx']
faro['neck_chest_dx'] = faro['Neck_x'] - faro['Mid_chest_upper_x']
faro['neck_chest_dy'] = faro['Neck_y'] - faro['Mid_chest_upper_y']
faro['neck_chest_dz'] = faro['Neck_z'] - faro['Mid_chest_upper_z']
faro['sb_neck_angle_X'] = faro['sb_neck_dy']/faro['neck_chest_dz']
faro['neck_chest_angle_Y'] = faro['neck_chest_dx']/faro['neck_chest_dz']
faro['chest_pelvis_dx'] = faro['Mid_chest_upper_x'] - faro['Pubis_x']
faro['chest_pelvis_dy'] = faro['Mid_chest_upper_y'] - faro['Pubis_y']
faro['chest_pelvis_dz'] = faro['Mid_chest_upper_z'] - faro['Pubis_z']
faro['chest_pelvis_angle_Y'] = faro['chest_pelvis_dx']/faro['chest_pelvis_dz']
faro['belt_dx'] = faro['Mid_chest_upper_x'] - faro['Mid_chest_lower_x']
faro['belt_dz'] = faro['Mid_chest_upper_z'] - faro['Mid_chest_lower_z']
#faro['lap_belt_fit'] = dataset.table.loc[faro.index, 'Lap_belt_fit']
faro['lap_belt_fit'] = faro[['LB_left_down_x', 'LB_right_down_x']].mean(axis=1)-faro['Pubis_x']
faro['pelvis_angle'] = 90 - np.degrees(np.arctan(faro['thigh_angle_Y'])) + np.degrees(-np.arctan(faro['chest_pelvis_angle_Y']))


# get the residual 
#x = sm.add_constant(faro['neck_pubis_dx'])
#y = dataset.table.loc[x.index, 'Knee_excursion']-faro.loc[x.index, 'Knee_x']
#ols = sm.OLS(y, x)
#rr = ols.fit()


min_seat_limits = faro.groupby('Model').min()['Seat_limit_x']

faro['Seat_length'] = faro[['Model','Seat_limit_x']].apply(lambda x: min_seat_limits[x['Model']], axis=1)
faro['diff_seat_length'] = faro['Seat_limit_x'] - faro['Seat_length'] + 1

#%% find predictors
ch = 'Chest_3ms'
#ch = 'Min_12CHST0000Y7ACXC'

corr_fun = 'kendall'
subset = dataset.table.index
drop = ['LB_left_up_y']


x = faro.loc[subset].select_dtypes(np.float64).dropna(axis=1, how='all').drop(drop, axis=1)
if ch in dataset.table:
    y = dataset.table.loc[subset, ch].to_frame()
else:
    y = dataset.features.loc[subset, ch].to_frame()

x, y = preprocess_data(x, y, missing_x='mean', missing_y='drop')

model = iter([Lars(n_nonzero_coefs=i) for i in range(1, 10)])
#eval_model = partial(SMRegressionWrapper, model=sm.OLS)
eval_model = partial(SMRegressionWrapper, model=sm.OLS)
vs = VariableSelector(x, y, model, eval_model=eval_model, corr_fun=corr_fun, incr_thresh=0.01, corr_thresh=0.6)
vs.find_variables()
vs.plot_variables()
print(vs.eval_results.rr.summary())
data = pd.concat((faro, dataset.table, dataset.features), axis=1)
data = data.loc[:,~data.columns.duplicated()]
for col in vs.predictors:
#for col in dataset.table:
    fig, ax = plt.subplots()
    sns.scatterplot(x=col, y=ch, hue='Model', data=data)

#%% plot 3d
import plotly_express as px
from plotly.offline import plot

subset = ['SE18-0054_3','SE18-0054_5']
points = ['Neck','Mid_chest_upper','SB_inboard']
data = pd.DataFrame({'x': faro.loc[subset,[i + '_x' for i in points]].values.flatten(),
                     'y': -faro.loc[subset,[i + '_y' for i in points]].values.flatten(),
                     'z': faro.loc[subset,[i + '_z' for i in points]].values.flatten(),
                     'hue': np.repeat(subset, len(points))})
fig = px.scatter_3d(x='x', y='y', z='z', color='hue', data_frame=data)
plot(fig)

#%% test things with interactions
import statsmodels.formula.api as smf
predictors = ['LB_left_down_x', 'LB_left_down_z']
x = faro[predictors]
x.columns = ['x' + str(i) for i in range(1,len(predictors)+1)]
if ch in dataset.table:
    y = dataset.table.loc[subset, ch].to_frame()
else:
    y = dataset.features.loc[subset, ch].to_frame()
y.columns = ['y']
    
model = smf.ols(formula='y ~ x1 + x2 + x1*x2', data = pd.concat((x,y), axis=1))
rr = model.fit()
print(rr.summary())
#%% print the original range for each model
ypred = pd.Series(vs.eval_results.rr.predict(), name='ypred', index=dataset.table.index)
regression_results = pd.concat((dataset.table['Model'], y, ypred), axis=1)
regression_results['yerr'] = ypred-y.squeeze()
grouped = list(regression_results.groupby('Model'))
grouped.append(('All', regression_results))

print('{0} with predictors {1}'.format(y.squeeze().name, vs.predictors))+3.25
for grp in grouped:
    act_min = grp[1][y.squeeze().name].min()
    act_max = grp[1][y.squeeze().name].max()
    pred_min = grp[1]['ypred'].min()
    pred_max = grp[1]['ypred'].max()
    worst_error = (grp[1][y.squeeze().name]-grp[1]['ypred']).abs().max()
    
    print(grp[0])
    print('Actual variance: {0}'.format(grp[1][y.squeeze().name].var()))
    print('Explained variance: {0}'.format(grp[1]['ypred'].var()))
    print('Unexplained variance: {0}'.format(grp[1][y.squeeze().name].var()-grp[1]['ypred'].var()))
    print('Actual range: {0} ({1}-{2})'.format(act_max-act_min, act_min, act_max))
    print('Error range: {0} ({1}-{2})'.format(grp[1]['yerr'].max()-grp[1]['yerr'].min(), grp[1]['yerr'].min(), grp[1]['yerr'].max()))
#    print('Predicted range: {0} ({1}-{2})'.format(pred_max-pred_min, pred_min, pred_max))
#    print('Unexplained range: {0}'.format((act_max-act_min)-(pred_max-pred_min)))
    print('\n')

#%%
for ch in dataset.timeseries.columns:
    x = {'2, 4': dataset.timeseries.loc[['SE18-0054_2', 'SE18-0054_4'], ch].apply(lambda x: x[100:600]),
         '3, 5': dataset.timeseries.loc[['SE18-0054_3','SE18-0054_5'], ch].apply(lambda x: x[100:600])}
    fig, ax = plt.subplots()
    ax = plot_overlay(ax, dataset.t[100:600], x)
    ax = set_labels(ax, {'title': ch, 'legend': {}})
    
