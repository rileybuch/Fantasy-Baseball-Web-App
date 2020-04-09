#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import bokeh
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from bokeh.plotting import figure, show
from bokeh import io
from bokeh.layouts import column, gridplot


# In[2]:


def create_res_plot(y_model, y_true, name='some text'):
    """
    This function plots the residuals for test data given the actual and predicted data.
    :param y_model: the predicted values from the model
    :param y_true: the actual values
    :param name: the stat being modeled
    :return:
    """
    p = figure(title=f'actual vs residuals for {name}', plot_width=600)
    p.xaxis.axis_label = 'predicted values'
    p.yaxis.axis_label = 'actual values'
    p.circle(x=y_model, y=y_true)
    max_range = int(np.max(y_model))
    min_range = int(np.min(y_model))
    x = range(min_range, max_range + 5, 10)
    y = x
    p.line(x=x, y=y)
    return p


# In[3]:


def create_dev_plot(yrange, training_score, testing_score, name='Deviance Plot'):
    p = figure(plot_width=400, title=name)
    p.line(yrange, training_score, color='blue', line_width=2, legend='Training Data')
    p.line(yrange, testing_score, color='orange', line_width=2, legend='Test Data')
    return p


# In[4]:


def create_feat_imp_plot(sorted_names, sorted_importance, name='Feature Importance'):
    p = figure(title=name, plot_width=400, y_range=sorted_names)
    p.hbar(y=sorted_names, height=0.5, left=0, right=sorted_importance)
    return p


# In[5]:


def model_gbr(df, dependent_var, not_features, params, train_pct=0.7):
    combined_df = pd.DataFrame()
    # we'll join on 'Name' for now so need to include that in dependent_cols
    dependent_cols = [dependent_var, 'Name']
    for season in range(1996, 2017):
        dep_df = df[df.Season == (season + 2)][dependent_cols]
        pred_n1_df = df[df.Season == (season + 1)]
        pred_n2_df = df[df.Season == season]
        temp_1_combined_df = pred_n2_df.merge(pred_n1_df, on='Name', suffixes=('_n2', '_n1'))
        temp_combined_df = temp_1_combined_df.merge(dep_df, on='Name')
        combined_df = combined_df.append(temp_combined_df)

    combined_df.reset_index(inplace=True, drop=True)
    # let's build models and predict
    train_df = combined_df.sample(frac=train_pct, random_state=1)
    test_df = combined_df.drop(train_df.index)

    x_train = train_df.drop(columns=not_features)
    y_train = train_df[dependent_var]
    x_test = test_df.drop(columns=not_features)
    y_test = test_df[dependent_var]

    regressor = GradientBoostingRegressor(**params)
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # get training deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(regressor.staged_predict(x_test)):
        test_score[i] = regressor.loss_(y_test, y_pred)

    # get feature importance
    feature_importance = regressor.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx = sorted_idx[-10:]
    features = list(x_test.columns[sorted_idx])

    plots = [create_res_plot(y_pred, y_test, name=f'{dependent_var} prediction, MAE = {mae: .2f}, R2 = {r2 :2f}'),
             create_dev_plot(np.arange(params['n_estimators']) + 1, regressor.train_score_, test_score,
                             f'{dependent_var} Deviance Plot'),
             create_feat_imp_plot(features, feature_importance[sorted_idx], f'{dependent_var} Feature Importance')]

    print(f'Modeling {dependent_var}: MAE was {mae} and r2 was {r2}')
    return regressor, plots


# In[6]:


def model_gbr_norm(df, dependent_var, not_features, train_pct=0.7, n_estimators=2000, max_depth=3, learning_rate=0.05,
                   min_samples_leaf=1):
    norm_var = f'{dependent_var}pG'
    combined_df = pd.DataFrame()
    # we'll join on 'Name' for now so need to include that in dependent_cols and need 'G' since we're predicting by game
    dependent_cols = [dependent_var, 'G', 'Name']

    not_features.append(norm_var)
    not_features.append('G_y')
    for season in range(1996, 2018):
        dep_df = df[df.Season == (season + 2)][dependent_cols]
        pred_n1_df = df[df.Season == (season + 1)]
        pred_n2_df = df[df.Season == season]
        temp_1_combined_df = pred_n2_df.merge(pred_n1_df, on='Name', suffixes=('_n2', '_n1'))
        temp_combined_df = temp_1_combined_df.merge(dep_df, on='Name')
        combined_df = combined_df.append(temp_combined_df)

    # let's normalize the predicted stats by game
    combined_df[norm_var] = combined_df.apply(lambda row: row[dependent_var] / row.G, axis=1)

    combined_df.reset_index(inplace=True, drop=True)
    # let's build models and predict
    train_df = combined_df.sample(frac=train_pct)
    test_df = combined_df.drop(train_df.index)

    x_train = train_df.drop(columns=not_features)
    y_train = train_df[norm_var]
    x_test = test_df.drop(columns=not_features)
    y_test = test_df[norm_var]

    regressor = GradientBoostingRegressor(**params)
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # get training deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(regressor.staged_predict(x_test)):
        test_score[i] = regressor.loss_(y_test, y_pred)

    # get feature importance
    feature_importance = regressor.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx = sorted_idx[-10:]
    features = list(x_test.columns[sorted_idx])

    plots = [create_res_plot(y_pred, y_test, name=f'{dependent_var} prediction, MAE = {mae: .2f}, R2 = {r2 :2f}'),
             create_dev_plot(np.arange(params['n_estimators']) + 1, regressor.train_score_, test_score,
                             f'{dependent_var} Deviance Plot'),
             create_feat_imp_plot(features, feature_importance[sorted_idx], f'{dependent_var} Feature Importance')]

    print(f'Modeling {dependent_var} per game: MAE was {mae} and r2 was {r2}')
    return regressor, plots


# In[7]:


def filter_data(input_file):
    # load up the pitching data
    pitcher_df = pd.read_csv(input_file)

    # remove all of the columns that have nan values
    nan_cols = pitcher_df.columns[pitcher_df.isna().any()].tolist()
    filter_df = pitcher_df.drop(columns=nan_cols)

    # drop any columns you don't want here
    cols_to_remove = ['Age Rng']
    filter_df.drop(columns=cols_to_remove, inplace=True)

    return filter_df


# In[8]:


def filter_norm_data(input_file):
    # load up the pitching data
    pitcher_df = pd.read_csv(input_file)

    # remove all of the columns that have nan values
    nan_cols = pitcher_df.columns[pitcher_df.isna().any()].tolist()
    filter_df = pitcher_df.drop(columns=nan_cols)

    # drop any columns you don't want here
    cols_to_remove = ['Age Rng']
    filter_df.drop(columns=cols_to_remove, inplace=True)

    stats_to_norm = ['W', 'L', 'SV', 'SO', 'IP', 'HR', 'CG', 'ShO', 'H', 'R', 'ER', 'BB', 'IBB']
    for stat in stats_to_norm:
        filter_df[f'{stat}pG'] = filter_df.apply(lambda row: row[stat] / row.G, axis=1)
        filter_df.drop(columns=[stat], inplace=True)

    return filter_df


# In[88]:


def model_pitching(hyperparams, csv_file='../pitching_data_1996_2019.csv'):
    # load up the batting data
    filter_df = filter_data(csv_file)

    # setup the list of what to model
    dep_variables = ['W', 'SV', 'ERA', 'WHIP', 'SO', 'IP', 'L', 'HR']
    models = {}
    plots = {}

    for dep_var in dep_variables:
        print(f'modeling {dep_var}')
        not_features_list = ['Name', 'Season_n1', 'Team_n1', 'Season_n2', 'Team_n2', dep_var]

        # model and predict quality
        models[dep_var], plots[dep_var] = model_gbr(filter_df, dep_var, not_features_list, hyperparams)
        

    bokeh_output = r'/Users/rmbuch/Desktop/fantasy_baseball/output.html'
    io.output_file(bokeh_output)
    figures = []
    for var in dep_variables:
        for i in range(len(plots[var])):
            figures.append(plots[var][i])

    show(gridplot(figures, ncols=3, plot_height=300, merge_tools=False))
    
    # make predictions for 2020
    pred_n1_df = filter_df[filter_df.Season == 2018]
    pred_n2_df = filter_df[filter_df.Season == 2019]
    combined_df = pred_n2_df.merge(pred_n1_df, on='Name', suffixes=('_n2', '_n1'))
    combined_df.reset_index(inplace=True, drop=True)
    X = combined_df.drop(columns=['Name', 'Season_n1', 'Team_n1', 'Season_n2', 'Team_n2'])
    predictions = combined_df[['Name', 'key_mlbam_n1']].copy()
    
    for stat, model in models.items():
        print(f'making {stat} predictions for 2020')
        preds = model.predict(X)
        predictions[stat] = preds

        
    print(predictions.head())
    


# In[89]:


def model_pitching_normalized(hyperparams, csv_file='../pitching_data_1996_2019.csv'):
    # load up the pitching, filter it, and put it into a dataframe
    filter_df = filter_norm_data(csv_file)

    # setup the list of what to model
    dep_variables = ['WpG', 'SVpG', 'ERA', 'WHIP', 'SOpG', 'IPpG', 'LpG', 'HRpG']
    models = {}
    plots = {}

    for dep_var in dep_variables:
        print(f'modeling {dep_var}')
        not_features_list = ['Name', 'Season_n1', 'Team_n1', 'Season_n2', 'Team_n2', dep_var]
        # model and predict quality
        models[dep_var], plots[dep_var] = model_gbr(filter_df, dep_var, not_features_list, hyperparams)

    bokeh_output = r'/Users/rmbuch/Desktop/fantasy_baseball/output_normalized.html'
    io.output_file(bokeh_output)
    figures = []
    for var in dep_variables:
        for i in range(len(plots[var])):
            figures.append(plots[var][i])

    show(gridplot(figures, ncols=3, plot_height=300, merge_tools=False))
    
    # make predictions for 2020
    pred_n1_df = filter_df[filter_df.Season == 2018]
    pred_n2_df = filter_df[filter_df.Season == 2019]
    combined_df = pred_n2_df.merge(pred_n1_df, on='Name', suffixes=('_n2', '_n1'))
    combined_df.reset_index(inplace=True, drop=True)
    X = combined_df.drop(columns=['Name', 'Season_n1', 'Team_n1', 'Season_n2', 'Team_n2'])
    predictions = combined_df[['Name', 'key_mlbam_n1']].copy()
    
    for stat, model in models.items():
        print(f'making {stat} predictions for 2020')
        preds = model.predict(X)
        predictions[stat] = preds

        
    print(predictions.head())


# #### if __name__ == "__main__":
#     pitching_data = r'pitching_data_1996_2019.csv'
# 
#     # this models the 5 standard stats based off of the previous year
#     params = {'n_estimators': 1000, 'max_depth': 3, 'min_samples_split': 2, 'learning_rate': 0.01,
#               'min_samples_leaf': 1}
# 
#     # this models the 8 stats, based off the previous year
#     model_pitching(params, pitching_data)
# 
#     # this models the 8 stats by game, normalized (except ERA and WHIP), based off the previous year
#     model_pitching_normalized(params, pitching_data)
#     
#     

# In[ ]:




