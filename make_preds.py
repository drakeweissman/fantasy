import win_pred_model
import pandas as pd
from espn_api.football import League
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
import numpy as np
import shap

#Fetch league data

# Initialize league
league_id = 26347
espn_s2 = 'AECJTHUB5QQ41P4C5vinQpk7fGVA6h%2BnbM7tsN7mhlpWupwMWzVIKnKFd219nyX3Ss37wALT0z0fYoIOd9zieRZOE6I3nG%2BSSEUksFfA43gw8Hv3ywuj9PXh1fTxJlA9O%2FPfzY9GgfQH1OwPqQsmvWx0Zt7YOZKaBvy1ORbTZfgMfOZCVkqNYWMpBZzHCzAun99t%2FS3i24onjEXOch2vI9E%2Ff4y5%2BRBiE%2BaPaOlfnMTy1d3DbG1E%2FYqnZzNWbT3Yk3%2FFq7cLHbHTL1HF4Ouvgf6N'
swid = '{C3FE8278-A2E3-4D18-86D2-0154124A1F16}'
year = 2023  # Replace with the specific year you want

# Initialize the league for the specific year
league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid, debug=False)

# Get current standings
def get_current_standings(league,year):
    all_matchups = win_pred_model.get_matchup_scores(league, year)
    all_matchups = pd.DataFrame(all_matchups)
    all_matchups['home_team_win'] = (all_matchups['home_score'] > all_matchups['away_score']).astype(int)
    standings_df = win_pred_model.create_standings((all_matchups))
    max_week = standings_df['prior_to_week'].max()
    standings_df = standings_df[standings_df['prior_to_week'] == max_week]
    return standings_df
standings_df = get_current_standings(league,year)
print(standings_df)