import pandas as pd


def risk_calc_l(x, stat):
    if x.overall_risk == 0:
        return x[stat]*1.05
    elif x.overall_risk == 1:
        return x[stat]
    else:
        return x[stat] * 0.95


def risk_calc_h(x, stat):
    if x.overall_risk == 0:
        return x[stat]*0.95
    elif x.overall_risk == 1:
        return x[stat]
    else:
        return x[stat] * 1.05


def calc_risk(df, cols_to_keep, cols_to_expand):
    df = df[cols_to_keep]
    index_2020 = df['Season'] == 2020
    df_2020 = df[index_2020]
    df_hist = df[~index_2020]

    # create low and high predictions adjusted by overall_risk
    for stat, ascend in cols_to_expand.items():
        df_2020[f'{stat}_l'] = df_2020.apply(lambda row: risk_calc_l(row, stat), axis=1)
        df_2020[f'{stat}_h'] = df_2020.apply(lambda row: risk_calc_h(row, stat), axis=1)
        df_2020[f'{stat}rL'] = df_2020[f'{stat}_l'].rank(method='min', ascending=ascend)
        df_2020[f'{stat}rM'] = df_2020[stat].rank(method='min', ascending=ascend)
        df_2020[f'{stat}rH'] = df_2020[f'{stat}_h'].rank(method='min', ascending=ascend)

    return pd.concat([df_2020, df_hist])


if __name__ == "__main__":
    # BATTERS
    cols_to_keep = ['Season', 'Name', 'Team', 'HR', 'TB', 'R', 'RBI', 'SB', 'AVG', 'OBP', 'SLG', 'key_mlbam',
                    'overall_risk']
    cols_to_expand = {'HR': False, 'TB': False, 'R': False, 'RBI': False, 'SB': False, 'AVG': False, 'OBP': False,
                      'SLG': False}
    batting_df = pd.read_csv('../batting_data_1996_2020_with_risk.csv')
    new_batting_df = calc_risk(batting_df, cols_to_keep, cols_to_expand)
    new_batting_df.to_csv('../batting_1996_2020_risk_calc.csv', index=False)

    # PITCHERS
    cols_to_keep = ['Season', 'Name', 'Team', 'Age', 'W', 'L', 'ERA', 'SV', 'IP', 'HR', 'SO', 'WHIP', 'key_mlbam',
                    'overall_risk']
    cols_to_expand = {'W': False, 'L': True, 'ERA': True, 'SV': False, 'IP': False, 'HR': True, 'SO': False,
                      'WHIP': True}

    # starters
    starters = pd.read_csv('../pitching_data_starters_1996_2020_with_risk.csv')
    new_starters = calc_risk(starters, cols_to_keep, cols_to_expand)
    new_starters.to_csv('../starters_1996_2020_risk_calc.csv', index=False)

    # relievers
    relievers = pd.read_csv('../pitching_data_relievers_1996_2020_with_risk.csv')
    new_relievers = calc_risk(relievers, cols_to_keep, cols_to_expand)
    new_relievers.to_csv('../relievers_1996_2020_risk_calc.csv', index=False)
