import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np  # Add this import at the top

# Load the dataset
data = pd.read_csv('loss.csv')

# Preprocessing: Drop 'Company Name' and convert categorical variables to dummy variables
data = data.drop(columns=[
    'Company Name',
    'Date of Attack',
    'Company Size',
    'Attack Vector/Type of Breach',
    'Time to Patch/Contain (days)',
    'Industry Subsector',
])

# Clean Financial Loss column
data['Financial Loss'] = (
    data['Financial Loss']
    .str.replace('[\$,]', '', regex=True)  # Remove $ and commas
    .astype(float)  # Convert to float first
    .astype(int)    # Convert to integer
)

# Clean Annual Revenue column
data['Annual Revenue'] = (
    data['Annual Revenue']
    .str.replace('[\$,]', '', regex=True)  # Remove $ and commas
    .str.replace(' billion', '000', regex=True)  # Convert billion to millions
    .str.replace(' million', '', regex=True)  # Remove word 'million'
    .astype(float)  # Convert to float
    .mul(1_000_000)  # Multiply by 1,000,000 to get actual value
    .astype(np.int64)  # Convert to 64-bit integer to handle larger numbers
)

# Add debug prints to verify the conversion
print("\nSample Financial Losses:")
print(data['Financial Loss'].head())
print("\nSample Annual Revenues:")
print(data['Annual Revenue'].head())

data = pd.get_dummies(data, drop_first=True)

# Separate features (X) and target (y)
y = data['Financial Loss']
X = data.drop(columns=['Financial Loss'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the model and feature columns for future use
joblib.dump(model, 'loss_prediction_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')  # Save feature columns
print("Model training complete and saved as 'loss_prediction_model.pkl'.")
print("Feature columns saved as 'model_columns.pkl'.")

# Function to predict financial loss based on user input
def predict_loss(model, input_data):
    # Convert input data to DataFrame and align with training data columns
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Align with training data columns
    
    # Predict financial loss
    prediction = model.predict(input_df)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Load the trained model
    model = joblib.load('loss_prediction_model.pkl')
    feature_columns = joblib.load('model_columns.pkl')  # Load feature columns
    
    # Example input data
    example_input = {
        'Industry Subsector': 'Healthcare Finance',
        'Company Size': 'Large',
        'Annual Revenue': 3400 * 1_000_000,  # $3.4 billion
        'Number of Employees': 15000,
        'Geographic Region': 'North America',
        'Attack Vector/Type of Breach': 'Ransomware',
        'Time to Patch/Contain (days)': 35
    }
    
    # Predict financial loss
    predicted_loss = predict_loss(model, example_input)
    print(f"Predicted Financial Loss: ${predicted_loss:,.2f}")