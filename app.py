from flask import Flask, render_template
from pred_pipeline import run_pipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_predictions():
    matchups_with_predictions = run_pipeline('xgb_win_pred_model.sav')
    return render_template('predictions.html', predictions=matchups_with_predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)