from flask import Flask, render_template
from pred_pipeline import run_pipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_predictions():
    #In future, get matchups_with_predictions with relevant columns from SQL database
    matchups_with_predictions = run_pipeline()
    return render_template('predictions.html', predictions=matchups_with_predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)