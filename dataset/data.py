import pandas as pd
import numpy as np

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


df_fte = Source.download(_FiveThirtyEight, parse_dates=['date'])
fd_dfs = []
columns = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'shots1', 'shots2', 'shotsot1', 'shotsot2', 
	'fouls1', 'fouls2', 'corners1', 'corners2', 'yellow1', 'yellow2', 'red1', 'red2',
    'MaxH', 'AvgH', 'MaxD', 'AvgD', 'MaxA', 'AvgA', ] 
rename_dict = {
    'BbMxH': 'MaxH', 'BbMxD': 'MaxD', 'BbMxA': 'MaxA', 'BbAvH': 'AvgH', 'BbAvD': 'AvgD', 'BbAvA': 'AvgA',
    'HS': 'shots1', 'AS': 'shots2', 'HST': 'shotsot1', 'AST': 'shotsot2', 'HF': 'fouls1', 'AF': 'fouls2', 
    'HC': 'corners1', 'AC': 'corners2', 'HY': 'yellow1', 'AY': 'yellow2', 'HR': 'red1', 'AR': 'red2',
    'Max>2.5': 'maxover', 'Max<2.5': 'maxunder', 'Avg>2.5': 'avgover', 'Avg<2.5': 'avgunder',
    'BbMx>2.5': 'maxover', 'BbMx<2.5': 'maxunder', 'BbAv>2.5': 'avgover', 'BbAv<2.5': 'avgunder'}

for league in LEAGUES:
    for season in SEASONS:
        df_league = Source.download(
            _FootballData.format(season=season, league=league),
            parse_dates=['Date'],
        )
        df_league.rename(columns=rename_dict, inplace=True)
        if any([col not in df_league.columns for col in columns]):
            continue
        fd_dfs.append(df_league[columns])
df_fd = pd.concat(fd_dfs)


print(df_fte)
print(df_fd)


# Rename team names for merging
df_mapping = pd.read_csv('mapping.csv')
mapping = df_mapping.set_index('replace').to_dict()['replace_with']
df_fd.replace(mapping, inplace=True)
    
# Merge dataframes
lkeys = ['date', 'team1', 'team2']
rkeys = ['Date', 'HomeTeam', 'AwayTeam']
df_update = pd.merge(df_fte, df_fd, how='left', left_on=lkeys, right_on=rkeys)
print(df_update)