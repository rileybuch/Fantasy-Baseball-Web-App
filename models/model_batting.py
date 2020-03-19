import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from bokeh.plotting import figure, show
from bokeh import io
from bokeh.layouts import column


def create_res_plot(y_model, y_true, name='some text'):
    """
    This function plots the residuals for test data given the actual and predicted data.
    :param y_model: the predicted values from the model
    :param y_true: the actual values
    :param name: the stat being modeled
    :return:
    """
    p = figure(title=f'actual vs residuals for {name}', plot_height=300, plot_width=800)
    p.xaxis.axis_label = 'predicted values'
    p.yaxis.axis_label = 'actual values'
    p.circle(x=y_model, y=y_true)
    max_range = int(np.max(y_model))
    min_range = int(np.min(y_model))
    x = range(min_range, max_range+5, 10)
    y = x
    p.line(x=x, y=y)
    return p


def model_gbr(df, dependent_var, not_features, train_pct=0.7, n_estimators=2000, max_depth=3, learning_rate=0.05,
              min_samples_leaf=1):

    combined_df = pd.DataFrame()
    # we'll join on 'Name' for now so need to include that in dependent_cols
    dependent_cols = [dependent_var, 'Name']
    for season in range(1996, 2018):
        dep_df = df[df.Season == (season + 1)][dependent_cols]
        pred_df = df[df.Season == season]
        temp_combined_df = pred_df.merge(dep_df, on='Name')
        combined_df = combined_df.append(temp_combined_df)

    combined_df.reset_index(inplace=True, drop=True)
    # let's build models and predict
    train_df = combined_df.sample(frac=train_pct)
    test_df = combined_df.drop(train_df.index)

    x_train = train_df.drop(columns=not_features)
    y_train = train_df[f'{dependent_var}_y']
    x_test = test_df.drop(columns=not_features)
    y_test = test_df[f'{dependent_var}_y']

    regressor = GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf
    )
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plot = create_res_plot(y_pred, y_test, name=f'{dependent_var} prediction, MAE = {mae: .2f}, R2 = {r2 :2f}')
    print(f'Modeling {dependent_var}: MAE was {mae} and r2 was {r2}')
    return regressor, plot


def model_gbr_norm(df, dependent_var, not_features, train_pct=0.7, n_estimators=2000, max_depth=3, learning_rate=0.05,
              min_samples_leaf=1):

    norm_var = f'{dependent_var}pG'
    combined_df = pd.DataFrame()
    # we'll join on 'Name' for now so need to include that in dependent_cols and need 'G' since we're predicting by game
    dependent_cols = [dependent_var, 'G', 'Name']

    not_features.append(norm_var)
    not_features.append('G_y')
    for season in range(1996, 2018):
        dep_df = df[df.Season == (season + 1)][dependent_cols]
        pred_df = df[df.Season == season]
        temp_combined_df = pred_df.merge(dep_df, on='Name')
        combined_df = combined_df.append(temp_combined_df)

    # let's normalize the predicted stats by game
    combined_df[norm_var] = combined_df.apply(lambda row: row[f'{dependent_var}_y'] / row.G_y, axis=1)

    combined_df.reset_index(inplace=True, drop=True)
    # let's build models and predict
    train_df = combined_df.sample(frac=train_pct)
    test_df = combined_df.drop(train_df.index)

    x_train = train_df.drop(columns=not_features)
    y_train = train_df[norm_var]
    x_test = test_df.drop(columns=not_features)
    y_test = test_df[norm_var]

    regressor = GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf
    )
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plot = create_res_plot(y_pred, y_test, name=f'{dependent_var} per game prediction: MAE = {mae: .2f}, R2 = {r2 :2f}')
    print(f'Modeling {dependent_var} per game: MAE was {mae} and r2 was {r2}')
    return regressor, plot


def model_batting_normalized(input_file='../batting_data_1996_2019_expanded.csv'):
    # load up the batting data
    bat_df = pd.read_csv(input_file)

    # remove all of the columns that have nan values, this leaves 59
    nan_cols = bat_df.columns[bat_df.isna().any()].tolist()
    filter_df = bat_df.drop(columns=nan_cols)

    # drop any columns you don't want here
    cols_to_remove = ['Age Rng']
    filter_df.drop(columns=cols_to_remove, inplace=True)

    # setup the list of what to model
    dep_variables = ['HR', 'RBI', 'AVG', 'SB', 'R', 'OBP', 'TB', 'SLG']
    models = {}
    plots = {}

    # setup the modeling hyperparams
    trees = 500
    depth = 3
    rate = 0.03
    samp_leaf = 1

    for dep_var in dep_variables:
        print(f'modeling {dep_var} per game (if appropriate)')
        not_features_list = ['Season', 'Name', 'Team', f'{dep_var}_y']

        # model and predict quality
        if (dep_var == 'AVG') or (dep_var == 'OBP') or (dep_var == 'SLG'):
            models[dep_var], plots[dep_var] = model_gbr(filter_df, dep_var, not_features_list, n_estimators=trees,
                                                             max_depth=depth, learning_rate=rate,
                                                             min_samples_leaf=samp_leaf)
        else:
            models[dep_var], plots[dep_var] = model_gbr_norm(filter_df, dep_var, not_features_list, n_estimators=trees,
                                                         max_depth=depth, learning_rate=rate,
                                                         min_samples_leaf=samp_leaf)

    bokeh_output = r'C:\Users\jkarp\PycharmProjects\fantasy_baseball\output_normalized.html'
    io.output_file(bokeh_output)
    figures = []
    for var in dep_variables:
        figures.append(plots[var])

    show(column(*figures))


def model_batting(input_file='../batting_data_1996_2019_expanded.csv'):
    # load up the batting data
    bat_df = pd.read_csv(input_file)

    # remove all of the columns that have nan values, this leaves 59
    nan_cols = bat_df.columns[bat_df.isna().any()].tolist()
    filter_df = bat_df.drop(columns=nan_cols)

    # drop any columns you don't want here
    cols_to_remove = ['Age Rng']
    filter_df.drop(columns=cols_to_remove, inplace=True)

    # setup the list of what to model
    dep_variables = ['HR', 'RBI', 'AVG', 'SB', 'R', 'OBP', 'TB', 'SLG']
    models = {}
    plots = {}

    # setup the modeling hyperparams
    trees = 500
    depth = 3
    rate = 0.03
    samp_leaf = 1

    for dep_var in dep_variables:
        print(f'modeling {dep_var}')
        not_features_list = ['Season', 'Name', 'Team', f'{dep_var}_y']

        # model and predict quality
        models[dep_var], plots[dep_var] = model_gbr(filter_df, dep_var, not_features_list, n_estimators=trees,
                                                             max_depth=depth, learning_rate=rate,
                                                             min_samples_leaf=samp_leaf)

    bokeh_output = r'C:\Users\jkarp\PycharmProjects\fantasy_baseball\output.html'
    io.output_file(bokeh_output)
    figures = []
    for var in dep_variables:
        figures.append(plots[var])

    show(column(*figures))


if __name__ == "__main__":
    batting_data = r'C:\Users\jkarp\PycharmProjects\fantasy_baseball\batting_data_1996_2019_expanded.csv'

    # this models the 5 standard stats based off of the previous year
    model_batting(batting_data)

    # this models the 5 standard stats by game (except batting average), based off the previous year
    model_batting_normalized(batting_data)
