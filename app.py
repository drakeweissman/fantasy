from flask import Flask, render_template
#from pred_pipeline import run_pipeline
import sqlite3
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_predictions():
    # Create a connection to your database
    conn = sqlite3.connect('fantasy_predictions.db')

    # Query the database to get the matchups for this week
    query = """
    SELECT home_team, away_team, predicted_winner, predicted_prob
    FROM matchup_preds
    ORDER BY date_added DESC
    LIMIT 5
    """
    matchups_with_predictions = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return render_template('predictions.html', predictions=matchups_with_predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)