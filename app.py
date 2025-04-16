from flask import Flask, request, render_template
import pandas as pd
import joblib

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
    # Get form data
    input_data = {
        'Industry Subsector': request.form['Industry Subsector'],
        'Company Size': request.form['Company Size'],
        'Annual Revenue': int(request.form['Annual Revenue']),
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