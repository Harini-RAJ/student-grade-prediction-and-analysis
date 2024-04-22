# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:28:49 2024

@author: harin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Select features and target variable
features = ['age', 'Fedu', 'Medu', 'studytime', 'failures', 'G1', 'G2']
X = df[features]
y = df['G3']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate each model
def evaluate_model(model):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Evaluate each model
results = {}
for name, model in models.items():
    mse, mae, r2 = evaluate_model(model)
    results[name] = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'R-squared': r2}

# Display results
results_df = pd.DataFrame(results)
print(results_df)
