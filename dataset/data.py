import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from datetime import datetime

_FiveThirtyEight = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
_FootballData = 'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'

LEAGUES = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'F1', 'N1', 'SP1', 'P1', 'SP2']
SEASONS = ['1617', '1718', '1920', '2021', '2122']


class Source:
        
    @staticmethod
    def download(link: str, **pandas_kwargs) -> pd.DataFrame():
        df = pd.read_csv(link, **pandas_kwargs)
        return df


# df_fte = Source.download(_FiveThirtyEight, parse_dates=['date'])
# fd_dfs = []
# columns = [
#     'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'shots1', 'shots2', 'shotsot1', 'shotsot2', 
# 	'fouls1', 'fouls2', 'corners1', 'corners2', 'yellow1', 'yellow2', 'red1', 'red2',
#     'Max1', 'Avg1', 'MaxD', 'AvgD', 'Max2', 'Avg2'] 
# rename_dict = {
#     'BbMxH': 'Max1', 'BbMxD': 'MaxD', 'BbMxA': 'Max2', 'BbAvH': 'Avg1', 'BbAvD': 'AvgD', 'BbAvA': 'Avg2',
#     'MaxH': 'Max1', 'AvgH': 'Avg1', 'MaxA': 'Max2', 'AvgA': 'Avg2',
#     'HS': 'shots1', 'AS': 'shots2', 'HST': 'shotsot1', 'AST': 'shotsot2', 'HF': 'fouls1', 'AF': 'fouls2', 
#     'HC': 'corners1', 'AC': 'corners2', 'HY': 'yellow1', 'AY': 'yellow2', 'HR': 'red1', 'AR': 'red2',
#     'Max>2.5': 'maxover', 'Max<2.5': 'maxunder', 'Avg>2.5': 'avgover', 'Avg<2.5': 'avgunder',
#     'BbMx>2.5': 'maxover', 'BbMx<2.5': 'maxunder', 'BbAv>2.5': 'avgover', 'BbAv<2.5': 'avgunder'}

# for league in LEAGUES:
#     for season in SEASONS:
#         df_league = Source.download(
#             _FootballData.format(season=season, league=league),
#             parse_dates=['Date'],
#         )
#         df_league.rename(columns=rename_dict, inplace=True)
#         if any([col not in df_league.columns for col in columns]):
#             continue
#         fd_dfs.append(df_league[columns])
# df_fd = pd.concat(fd_dfs)

# # Rename team names for merging
# df_mapping = pd.read_csv('dataset\\mapping.csv')
# mapping = df_mapping.set_index('replace').to_dict()['replace_with']
# df_fd.replace(mapping, inplace=True)
    
# # Merge dataframes
# lkeys = ['date', 'team1', 'team2']
# rkeys = ['Date', 'HomeTeam', 'AwayTeam']
# df_final = pd.merge(df_fte, df_fd, how='left', left_on=lkeys, right_on=rkeys)

# df_final.to_csv('final.csv', index=False)

df_final = pd.read_csv('final.csv')
print(len(df_final))
df_final.dropna(subset=['xg1', 'xg2', 'shots1', 'shots2', 'Max1', 'MaxD', 'Max2'], inplace=True)
df_final.drop(['league', 'league_id', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'Date', 'HomeTeam', 'AwayTeam', 'FTR'], axis=1, inplace=True)
df_final['importance1'].fillna(df_final['importance1'].mean(), inplace=True)
df_final['importance2'].fillna(df_final['importance2'].mean(), inplace=True)

home_columns = df_final.columns[df_final.columns.str.contains('1')].to_list()
away_columns = df_final.columns[df_final.columns.str.contains('2')].to_list()
df_final = pd.concat([
        df_final.assign(home=1),
        df_final.assign(home=0).rename(
            columns=dict(zip(home_columns + away_columns, away_columns + home_columns))
        )
    ])

df_final.set_index(['date', 'team1', 'team2'], inplace=True)

test_df = df_final[df_final['season'] == 2021]
train_df = df_final[df_final['season'] != 2021][0:int(len(df_final[df_final['season'] != 2021])*0.8)]
val_df = df_final[df_final['season'] != 2021][int(len(df_final[df_final['season'] != 2021])*0.8):]
print(f'total: {len(df_final)}, train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}, check: {len(train_df) + len(val_df) + len(test_df)}')

train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# df_std = (df_final - train_mean) / train_std
# df_std = df_std.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# _ = ax.set_xticklabels(df_final.keys(), rotation=90)
# plt.show()

groups = train_df.groupby('team1')

def get_sequence(label_row, groups):
    team1 = label_row.Index[1]
    team2 = label_row.Index[2]
    date = label_row.Index[0]
    home_seq = groups.get_group(team1)
    home_seq = home_seq[home_seq.index.get_level_values('date') < date].tail(6)
    away_seq = groups.get_group(team2)
    away_seq = away_seq[away_seq.index.get_level_values('date') < date].tail(6)
    if len(home_seq) < 6 or len(away_seq) < 6:
        return
    else:
        home_seq.reset_index(inplace=True)
        home_seq.drop(['date', 'team1', 'team2', 'season'], axis=1, inplace=True)
        away_seq.reset_index(inplace=True)
        away_seq.drop(['date', 'team1', 'team2', 'season'], axis=1, inplace=True)
        seq = home_seq.join(away_seq, lsuffix='_home', rsuffix='_away')
        return seq

sequences = list()
for row in train_df.itertuples(index=True):
    sequence = get_sequence(row, groups)
    if sequence is not None:
        sequences.append(sequence)
