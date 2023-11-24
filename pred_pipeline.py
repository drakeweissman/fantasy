import pandas as pd
from espn_api.football import League
import xgboost as xgb
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pickle
from train_functions import matchups_preprocessing
from train_functions import baseline_pred
from predict_functions import get_current_standings
from predict_functions import get_current_matchups
from predict_functions import matchups_predict
from predict_functions import shap_preds

def run_pipeline(filename):
    #Fetch league data
    # Initialize league
    league_id = 26347
    espn_s2 = 'AECJTHUB5QQ41P4C5vinQpk7fGVA6h%2BnbM7tsN7mhlpWupwMWzVIKnKFd219nyX3Ss37wALT0z0fYoIOd9zieRZOE6I3nG%2BSSEUksFfA43gw8Hv3ywuj9PXh1fTxJlA9O%2FPfzY9GgfQH1OwPqQsmvWx0Zt7YOZKaBvy1ORbTZfgMfOZCVkqNYWMpBZzHCzAun99t%2FS3i24onjEXOch2vI9E%2Ff4y5%2BRBiE%2BaPaOlfnMTy1d3DbG1E%2FYqnZzNWbT3Yk3%2FFq7cLHbHTL1HF4Ouvgf6N'
    swid = '{C3FE8278-A2E3-4D18-86D2-0154124A1F16}'
    year = 2023  # Replace with the specific year you want

    # Initialize the league for the specific year
    league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid, debug=False)
    final_features = ['home_team_win_pct', 'away_team_win_pct','home_team_ppg','away_team_ppg', 'season','away_projected','home_projected','week']

    # Get current standings
    standings_df = get_current_standings(league,year)

    # Get new matchups
    current_matchups = get_current_matchups(league,year)

    #Matchups Preprocessing to incorporate season stats
    matchups_cleaned = matchups_preprocessing(current_matchups,standings_df)

    #Load the model and make predictions
    #filename = 'xgb_win_pred_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    #Make matchups predictions
    matchups_with_predictions = matchups_predict(matchups_cleaned,loaded_model,final_features)
    matchups_with_predictions = matchups_with_predictions[['home_team','away_team','predicted_winner','predicted_prob']]
    return matchups_with_predictions

# Use shap to check feature importance for each of the 5 predictions in the current week
#shap_preds(matchups_cleaned,final_features,loaded_model)


