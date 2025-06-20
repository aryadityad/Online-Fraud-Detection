# ==============================================================================
# app.py
#
# Flask application to serve the fraud detection model via a web interface.
# It loads the pre-trained model and provides predictions based on user input.
# ==============================================================================
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model pipeline
try:
    with open('fraud_detection_pipeline.pkl', 'rb') as file:
        model_pipeline = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'fraud_detection_pipeline.pkl' not found. Please run the training script first.")
    model_pipeline = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model_pipeline = None

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles prediction requests.
    - GET: Renders the prediction form.
    - POST: Processes form data, makes a prediction, and returns the result.
    """
    if request.method == 'POST':
        if model_pipeline is None:
            return "Model is not loaded. Please check the server logs.", 500

        try:
            # Retrieve data from the form
            step = float(request.form['step'])
            transaction_type = request.form['type']
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])

            # --- Feature Engineering (must exactly match the training script) ---
            hour_of_day = step % 24
            balance_diff_orig = oldbalanceOrg - newbalanceOrig
            balance_diff_dest = newbalanceDest - oldbalanceDest
            amount_log = np.log1p(amount)

            # Create a DataFrame from the input in the correct order for the model
            input_data = pd.DataFrame([[
                transaction_type, oldbalanceOrg, newbalanceOrig, oldbalanceDest,
                newbalanceDest, hour_of_day, balance_diff_orig,
                balance_diff_dest, amount_log
            ]], columns=[
                'type', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                'newbalanceDest', 'hour_of_day', 'balance_diff_orig',
                'balance_diff_dest', 'amount_log'
            ])

            # Make a prediction using the loaded pipeline
            prediction = model_pipeline.predict(input_data)
            prediction_result = int(prediction[0])

            # Render the result on the submit page
            return render_template('submit.html', prediction=prediction_result)

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return f"An error occurred during prediction. Please check the inputs. Error: {e}", 400

    # If the request method is GET, just render the prediction form
    return render_template('predict.html')

if __name__ == '__main__':
    # Runs the Flask application in debug mode
    app.run(debug=True)
