import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from bokeh.plotting import figure, show
from bokeh import io


def create_res_plot(y_model, y_true, name='some text'):
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


# load up the batting data
df = pd.read_csv("../batting_data_1996_2019.csv")

# remove all of the columns that have nan values, this leaves 59
nan_cols = df.columns[df.isna().any()].tolist()
filter_df = df.drop(columns=nan_cols)

# drop any columns you don't want here
cols_to_remove = ['Age Rng']
filter_df.drop(columns=cols_to_remove, inplace=True)

# What are we modeling?  Let's just do an example for modeling 2001 based on 2000
dependent_cols = ['HR']
not_features = ['Season', 'Name', 'Team', 'HR_y']
combined_df = pd.DataFrame()

for season in range(1996, 2018):
    dep_df = filter_df[filter_df.Season == (season+1)][['Name', 'HR']]
    pred_df = filter_df[filter_df.Season == season]
    temp_combined_df = pred_df.merge(dep_df, on='Name')
    combined_df = combined_df.append(temp_combined_df)


# let's build models and predict
combined_df.reset_index(inplace=True, drop=True)
train_pct = 0.7
train_df = combined_df.sample(frac=train_pct)
test_df = combined_df.drop(train_df.index)

x_train = train_df.drop(columns=not_features)
y_train = train_df.HR_y
x_test = test_df.drop(columns=not_features)
y_test = test_df.HR_y

regressor = GradientBoostingRegressor(
    max_depth=3,
    n_estimators=500,
    learning_rate=0.1,
    min_samples_leaf=1
)

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plot = create_res_plot(y_pred, y_test, name=f'HR prediction, MAE = {mae: .2f}, R2 = {r2 :2f}')
bokeh_output = r'C:\Users\jkarp\PycharmProjects\fantasy_baseball\output.html'
io.output_file(bokeh_output)
show(plot)

print(f'MAE was {mae} and r2 was {r2}')

