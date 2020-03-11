import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh import io
from bokeh.palettes import d3 as palette

def create_res_plot(y_pred, y_true, name='some text'):
    p = figure(title=f'actual vs residuals for {name}', plot_height=300, plot_width=800)
    p.xaxis.axis_label = 'predicted values'
    p.yaxis.axis_label = 'actual values'
    p.circle(x=y_pred, y=y_true)
    max = int(np.max(y_pred))
    min = int(np.min(y_pred))
    x = range(min, max+5, 10)
    y = x
    p.line(x=x, y=y)
    return p

# load up the batting data
df = pd.read_csv("../batting_data_1996_2019.csv")

# remove all of the columns that have nan values, this leaves 59
nan_cols = df.columns[df.isna().any()].tolist()
filter_df = df.drop(columns=nan_cols)

# drop any columns you don't want here
cols_to_remove = ['Age Rng']
filter_df.drop(columns=cols_to_remove, inplace=True)

seasons = filter_df.Season.unique()

# this is super-slow, removing it for now
# season_dict = {}
# for season in seasons:
#     temp_df = filter_df[filter_df.Season==season]
#     temp_df.reset_index(inplace=True, drop=True)
#     season_dict[season] = temp_df

# What are we modeling?  Let's just do an example for modeling 2001 based on 2000
dependent_cols = ['HR']
dep_01_df = filter_df[filter_df.Season==2001][['Name', 'HR']]

df_2000 = filter_df[filter_df.Season==2000]

combined_df = df_2000.merge(dep_01_df, on='Name')

x_train = combined_df.copy()
not_features = ['Season', 'Name', 'Team', 'HR_y']
x_train.drop(columns=not_features, inplace=True)
y_train = combined_df.HR_y

regressor = GradientBoostingRegressor(
    max_depth = 3,
    n_estimators=1000,
    learning_rate=0.05,
    min_samples_leaf=1
)
regressor.fit(x_train, y_train)

# let's try it out on 2002 data based on 2001 data...
dep_02_df = filter_df[filter_df.Season==2002][['Name', 'HR']]
df_2001 = filter_df[filter_df.Season==2001]
combined_df_test = df_2001.merge(dep_02_df, on='Name')

x_test = combined_df_test.copy()
x_test.drop(columns=not_features, inplace=True)
y_test = combined_df_test.HR_y

y_pred = regressor.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plot = create_res_plot(y_pred, y_test, name = f'HR prediction, MAE = {mae: .2f}, R2 = {r2 :2f}')
bokeh_output = r'C:\Users\jkarp\PycharmProjects\fantasy_baseball\output.html'
io.output_file(bokeh_output)
show(plot)

print(f'MAE was {mae} and r2 was {r2}')

print("waiting")