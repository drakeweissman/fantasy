import pandas as pd
from espn_api.football import League
import xgboost as xgb
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
import shap
from train_functions import get_matchup_scores
from train_functions import create_standings

def get_current_standings(league,year):
    all_matchups = get_matchup_scores(league, year)
    all_matchups = pd.DataFrame(all_matchups)
    all_matchups['home_team_win'] = (all_matchups['home_score'] > all_matchups['away_score']).astype(int)
    standings_df = create_standings((all_matchups))
    max_week = standings_df['prior_to_week'].max()
    standings_df = standings_df[standings_df['prior_to_week'] == max_week]
    return standings_df

def get_current_matchups(league, year):
    all_matchups = get_matchup_scores(league, year)
    matchups_df = pd.DataFrame(all_matchups)
    max_week = matchups_df['week'].max()
    matchups_df = matchups_df[matchups_df['week'] == max_week]
    return matchups_df

def matchups_predict(matchups_cleaned,loaded_model,final_features):
    # Limit to just model features
    matchups_features = matchups_cleaned[final_features]
    # Use the loaded model to make predictions
    probabilities = loaded_model.predict_proba(matchups_features)
    labels = loaded_model.predict(matchups_features)

    # Create a new DataFrame with the predicted labels and probabilities
    predictions_df = pd.DataFrame({'home_team_win_pred': labels, 'home_team_win_prob': probabilities[:,1]})

    # Concatenate the predictions DataFrame with the original matchups_cleaned DataFrame
    matchups_with_predictions = pd.concat([matchups_cleaned, predictions_df], axis=1)

    # Add a predicted_team column based on home_team_win_pred
    matchups_with_predictions['predicted_winner'] = np.where(matchups_with_predictions['home_team_win_pred'] == 1, matchups_with_predictions['home_team'], matchups_with_predictions['away_team'])
    return matchups_with_predictions

def shap_preds(matchups_cleaned,final_features,loaded_model):
    matchups_features = matchups_cleaned[final_features]
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(matchups_features)
    shap.initjs()

    # Iterate through each row in the DataFrame and generate a force plot for each row
    for i in range(len(matchups_cleaned)):
        display(shap.force_plot(explainer.expected_value, shap_values[i,:], matchups_features.iloc[i,:]))