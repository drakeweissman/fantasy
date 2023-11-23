import pandas as pd
from espn_api.football import League
import xgboost as xgb
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
import shap

#Get all matchup scores function
def get_matchup_scores(league, year):
    matchup_scores = []

    # Iterate through each week
    for week in range(1, league.current_week + 1):
        # Get the box scores for the current week
        box_scores = league.box_scores(week)

        # Iterate through each game in the box scores
        for game_id, game in enumerate(box_scores, start=1):
            home_team = game.home_team
            away_team = game.away_team

            # Retrieve the scores for each team in the game
            home_score = game.home_score
            away_score = game.away_score

            # Additional information
            season = year
            home_projected = game.home_projected
            away_projected = game.away_projected 

            matchup_info = {
                "game_id": game_id,
                "season": season,
                "week": week,
                "home_team": home_team.team_name,
                "home_team_id": home_team.team_id,
                "home_score": home_score,
                "away_team": away_team.team_name,
                "away_team_id": away_team.team_id,
                "away_score": away_score,
                "home_projected": home_projected,
                "away_projected": away_projected,
            }

            matchup_scores.append(matchup_info)

    return matchup_scores

#Training Data Specific Cleaning
def training_cleaning(matchup_scores):
    matchups_df = pd.DataFrame(matchup_scores)
    matchups_df['home_team_win'] = (matchups_df['home_score'] > matchups_df['away_score']).astype(int)
    max_week = matchups_df['week'].max()
    matchups_df = matchups_df[matchups_df['week'] != max_week]
    return matchups_df

#Create a historical standings table
def create_standings(matchups_df):
    # Create an empty DataFrame to store the standings
    standings_df = pd.DataFrame()
    # Get a list of all unique team IDs
    team_ids = matchups_df['home_team_id'].unique()
    # Determine the maximum number of weeks in the dataset
    max_week = matchups_df['week'].max()
    # Iterate through each team
    for team_id in team_ids:
        # Create a DataFrame for the current team with all weeks' statistics
        team_df = pd.DataFrame({
            'team_id': [team_id] * max_week,
            'prior_to_week': list(range(1, max_week + 1)),
            'wins': 0,
            'losses': 0,
            'points_for': 0,
            'points_against': 0,
            'win_percentage': 0,
            'points_per_game': 0,
            'points_against_per_game': 0
        })

        # Iterate through each week
        for week in range(1, max_week + 1):
            # Filter the DataFrame to get data prior to the current week
            prior_to_week_df = matchups_df[matchups_df['week'] < week]

            # Filter the DataFrame to get matches involving the current team
            team_matches = prior_to_week_df[(prior_to_week_df['home_team_id'] == team_id) | (prior_to_week_df['away_team_id'] == team_id)]

            # Calculate team statistics
            team_wins = sum(team_matches['home_team_id'] == team_id)
            team_losses = sum(team_matches['away_team_id'] == team_id)
            team_points_for = sum(team_matches.loc[team_matches['home_team_id'] == team_id, 'home_score']) + sum(team_matches.loc[team_matches['away_team_id'] == team_id, 'away_score'])
            team_points_against = sum(team_matches.loc[team_matches['home_team_id'] == team_id, 'away_score']) + sum(team_matches.loc[team_matches['away_team_id'] == team_id, 'home_score'])
            total_games = team_wins + team_losses
            win_percentage = team_wins / total_games if total_games > 0 else 0
            points_per_game = team_points_for / total_games if total_games > 0 else 0
            points_against_per_game = team_points_against / total_games if total_games > 0 else 0

            # Update the current week's statistics in the team's DataFrame
            team_df.loc[week - 1, 'wins'] = team_wins
            team_df.loc[week - 1, 'losses'] = team_losses
            team_df.loc[week - 1, 'points_for'] = team_points_for
            team_df.loc[week - 1, 'points_against'] = team_points_against
            team_df.loc[week - 1, 'win_percentage'] = win_percentage
            team_df.loc[week - 1, 'points_per_game'] = points_per_game
            team_df.loc[week - 1, 'points_against_per_game'] = points_against_per_game

        # Append the team's DataFrame to the standings DataFrame
        standings_df = pd.concat([standings_df, team_df], ignore_index=True)
    return standings_df

#Matchups Preprocessing to incorporate season stats
def matchups_preprocessing(matchups_df,standings_df):
    #Column creation and set index
    matchups_df['matchup_id'] =  matchups_df['season'].astype(str) + matchups_df['week'].astype(str) + matchups_df['game_id'].astype(str)
    matchups_df.set_index('matchup_id', inplace=True)

    # Merge historical standings into matchup data to get team stats prior to each matchup

    # Merge 'standings_df' into 'df' for home team's statistics
    matchups_df = pd.merge(matchups_df, standings_df, how='left', left_on=['home_team_id', 'week'], right_on=['team_id', 'prior_to_week'])
    # Rename the columns for home team's statistics
    matchups_df = matchups_df.rename(columns={
        'win_percentage': 'home_team_win_pct',
        'points_per_game': 'home_team_ppg'
    })
    # Drop the redundant columns from the merge
    matchups_df = matchups_df.drop(['team_id', 'prior_to_week', 'wins', 'losses', 'points_for', 'points_against', 'points_against_per_game'], axis=1)
    # Merge 'standings_df' into 'df' for away team's statistics
    matchups_df = pd.merge(matchups_df, standings_df, how='left', left_on=['away_team_id', 'week'], right_on=['team_id', 'prior_to_week'])
    # Rename the columns for away team's statistics
    matchups_df = matchups_df.rename(columns={
        'win_percentage': 'away_team_win_pct',
        'points_per_game': 'away_team_ppg'
    })
    # Drop the redundant columns from the merge
    matchups_df = matchups_df.drop(['team_id', 'prior_to_week', 'wins', 'losses', 'points_for', 'points_against', 'points_against_per_game'], axis=1)
    return matchups_df

def feature_importances(model, X_train,N):
    feature_importances = model.feature_importances_

    # Get the names of the features
    feature_names = X_train.columns

    # Create a DataFrame to display the feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Display the top N most important features
    N = 7  # Change N to the number of top features you want to display
    top_features = importance_df.head(N)


    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(N), top_features['Importance'], align='center')
    plt.yticks(range(N), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances')
    plt.show()
    
def baseline_pred(dataset):
    #if home team has higher projection, predict home team win
    dataset['home_team_win_pred'] = (dataset['home_projected'] > dataset['away_projected']).astype(int)
    #check accuracy of baseline model
    dataset['home_team_win'] = (dataset['home_score'] > dataset['away_score']).astype(int)
    dataset['home_team_win_correct'] = (dataset['home_team_win'] == dataset['home_team_win_pred']).astype(int)
    baseline_accuracy = dataset['home_team_win_correct'].mean()
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")