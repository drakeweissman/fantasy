# Applying Continuous Learning to Predict Fantasy Football Matchups

Goal: Build an ML system that applies continuous learning to automatically retrain + deploy a new model each week to predict the results of the matchups in my Fantasy Football league

![App Example](Matchups%20Example.png)

## Architecture

The system is powered by two weekly workflows that run via Github Actions:

1. **Retrain Workflow:**
   - Fetches new league data using the ESPN web scraping API.
   - Utilizes all historical data to retrain an XGBoost algorithm.
   - Saves the retrained model as a new pickle file.

2. **Predict Workflow:**
   - Fetches current matchups using the ESPN web scraping API.
   - Uses the latest pickle file to make predictions.
   - Saves the predictions to the predictions database.

The `app.py` file reads the current matchups and predictions from the database and displays them in the frontend.

## Quick Start

1. **Install Docker:**

   Make sure you have [Docker](https://www.docker.com/products/docker) installed on your machine.

2. **Pull the Docker Image:**

   ```bash
   docker pull drakeweissman/fantasy-app

3. **Run the Docker Image:**

   ```bash
    docker run -p 5000:5000 drakeweissman/fantasy-app
   
4. **Access the Flask App:**

  Open a web browser and go to http://localhost:5000 to interact with the Flask app.


## Additional Information
If you encounter any issues or have questions, feel free to open an issue.
Contributions are welcome! If you'd like to contribute to the development of this app, please fork the repository and submit a pull request.

Note: Retraining has stopped now that the season is finished, but will begin again next season
