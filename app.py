from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('loss_prediction_model.pkl')

# Load the feature columns used during training
X_columns = joblib.load('model_columns.pkl')  # Load saved feature columns

# Initialize Flask app
app = Flask(__name__)

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data and convert Annual Revenue to proper format
    annual_revenue = (
        pd.Series(request.form['Annual Revenue'])
        .str.replace('[\$,]', '', regex=True)  # Remove $ and commas
        .str.replace(' billion', '000', regex=True)  # Convert billion to millions
        .str.replace(' million', '', regex=True)  # Remove word 'million'
        .astype(float)  # Convert to float
        .mul(1_000_000)  # Multiply by 1,000,000 to get actual value
        .astype(np.int64)  # Convert to 64-bit integer
    ).iloc[0]  # Get the single value
    
    input_data = {
        'Industry Subsector': request.form['Industry Subsector'],
        'Company Size': request.form['Company Size'],
        'Annual Revenue': annual_revenue,
        'Number of Employees': int(request.form['Number of Employees']),
        'Geographic Region': request.form['Geographic Region'],
        'Attack Vector/Type of Breach': request.form['Attack Vector/Type of Breach'],
        'Time to Patch/Contain (days)': int(request.form['Time to Patch/Contain (days)'])
    }

    # Convert input data to DataFrame and align with training data columns
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X_columns, fill_value=0)

    # Predict financial loss
    prediction = model.predict(input_df)[0]

    # Return the result
    return render_template('result.html', prediction=f"${prediction:,.2f}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)