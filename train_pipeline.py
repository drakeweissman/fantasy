import pandas as pd
from espn_api.football import League
import xgboost as xgb
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pickle
from train_functions import get_matchup_scores
from train_functions import training_cleaning
from train_functions import create_standings
from train_functions import matchups_preprocessing
from train_functions import feature_importances
from train_functions import baseline_pred

#Fetch league data
# Initialize league
league_id = 26347
espn_s2 = 'AECJTHUB5QQ41P4C5vinQpk7fGVA6h%2BnbM7tsN7mhlpWupwMWzVIKnKFd219nyX3Ss37wALT0z0fYoIOd9zieRZOE6I3nG%2BSSEUksFfA43gw8Hv3ywuj9PXh1fTxJlA9O%2FPfzY9GgfQH1OwPqQsmvWx0Zt7YOZKaBvy1ORbTZfgMfOZCVkqNYWMpBZzHCzAun99t%2FS3i24onjEXOch2vI9E%2Ff4y5%2BRBiE%2BaPaOlfnMTy1d3DbG1E%2FYqnZzNWbT3Yk3%2FFq7cLHbHTL1HF4Ouvgf6N'
swid = '{C3FE8278-A2E3-4D18-86D2-0154124A1F16}'
year = 2023  # Replace with the specific year you want

# Initialize the league for the specific year
league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid, debug=False)


#Get all matchup scores
matchup_scores = get_matchup_scores(league, year)

#Training Data Specific Cleaning
matchups_df = training_cleaning(matchup_scores)

#Create a historical standings table
standings_df = create_standings(matchups_df)

#Matchups Preprocessing to incorporate season stats
matchups_df = matchups_preprocessing(matchups_df,standings_df)

# ### Training

# Specify the features and target variable
final_features = ['home_team_win_pct', 'away_team_win_pct','home_team_ppg','away_team_ppg', 'season','away_projected','home_projected','week']
X = matchups_df[final_features] # Features
y = matchups_df['home_team_win']

#Split data based into training and test sets (Weeks 1-6, Weeks 7-9)) #Will make this a function at some point
total_weeks = X['week'].nunique()
cutoff_week = int(total_weeks * 0.91)

X_train = X[X['week'] < cutoff_week]
X_test = X[X['week'] >= cutoff_week]
y_train = y[X_train.index]
y_test = y[X_test.index]

#Fit model
# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)

#Calc Model Metrics

# Calculate accuracy on training and test data
test_accuracy = xgb_classifier.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get probability estimates for both training and test data
y_train_prob = xgb_classifier.predict_proba(X_train)
y_test_prob = xgb_classifier.predict_proba(X_test)

# Calculate log loss on training and test data
test_log_loss = log_loss(y_test, y_test_prob)
print(f"Test Log Loss: {test_log_loss:.4f}")

# Baseline Model Using ESPN Projections
baseline_pred(training_cleaning(get_matchup_scores(league, year)))

# Check feature importances
feature_importances(xgb_classifier,X_train, 7)

# #Pickle the model
# filename = 'xgb_win_pred_model.sav'
# pickle.dump(xgb_classifier, open(filename, 'wb'))

