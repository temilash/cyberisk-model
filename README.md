Cyber Risk Financial Loss Predictor
Overview
This project is a machine learning-powered web application that predicts the potential financial loss a company might incur from a cyber attack. It uses historical cyber incident data and company attributes to estimate the loss based on user input.

Features
Interactive Web Form: Users can input company details and cyber incident characteristics via a professional, user-friendly interface.
Dropdown Menus: For categorical fields like Company Size and Geographic Region, dropdowns ensure valid input.
Flexible Revenue Input: Annual Revenue can be entered in formats like "$2,500 million" or "3.4 billion".
Machine Learning Model: A trained Random Forest regression model predicts financial loss based on the provided inputs.
Feature Importance: The model highlights which factors most influence the predicted loss.
How It Works
Data Preparation:
The app uses a dataset (loss.csv) containing real-world cyber incidents, company financials, and outcomes. Data is cleaned and categorical variables are encoded for modeling.

Model Training:
The Random Forest model is trained on the processed data. Feature columns are saved for consistent input formatting.

Web Application:
Built with Flask, the app loads the trained model and feature columns. Users submit company and incident details, which are processed and passed to the model for prediction.

Prediction Output:
The predicted financial loss is displayed on a results page, formatted for clarity.
