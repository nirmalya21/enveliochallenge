# enveliochallenge
 coding challenge to predict current_A

Project Title: Energy Grid Current Forecasting

Table of Contents

Introduction

Data Description

Installation Requirements

Project Structure

Data Preprocessing

Model Development

Model Evaluation

Forecasting

Conclusion

Future Work

Introduction

This project aims to develop a predictive model for forecasting electrical current (current_A) in an energy grid using historical meteorological and current data. Various regression techniques, including XGBoost and Linear Regression, are utilized to enhance prediction accuracy.

Data Description

The dataset used in this project contains hourly data with the following columns:

**timestamp:** The date and time of the observation.

**temperature_C:** The ambient temperature in degrees Celsius.

**wind_speed_mps:** The wind speed in meters per second.

**solar_radiation_Wpm2:** The solar radiation measured in watts per square meter.

**current_A:** The target variable representing the electrical current in amperes.

**Installation Requirements**

To run this project, ensure that you have the following Python libraries installed:
numpy

pandas

matplotlib

seaborn

joblib

scikit-learn

xgboost

statsmodels

You can install the required libraries using pip:

pip install numpy pandas matplotlib seaborn joblib scikit-learn xgboost statsmodels

Project Structure
/project-directory
│
├── models/
│   ├── xgboost_model.json
│   └── scaler.pkl
│
├── data/
│   └── synthetic_energy_grid_data.csv
│   └── new_data.csv
├── notebooks/
│   └── model development.ipynb
│   └── model_inference (1).ipynb
│
└── main.py

**Data Preprocessing**

**Loading the Data:**
The dataset is loaded into a Pandas DataFrame with the timestamp parsed as dates.

**Handling Missing Values:** 
Infinite values are replaced with NaN and interpolated linearly. Rows with remaining NaN values are dropped.

**Feature Engineering:**
Lagged features for current_A, temperature_C, wind_speed_mps, and solar_radiation_Wpm2 are created for the past 24 hours.
Rolling mean features for current_A are computed over 3, 6, 12, and 24-hour windows.

**Model Development**
The following models are developed for forecasting:

**XGBoost Regressor:**
Hyperparameters are tuned using RandomizedSearchCV.
Best parameters obtained: {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 1.0}.

**Linear Regression:**
Standardized features are used for training the model.

**Model Evaluation**
The models are evaluated using Mean Absolute Error (MAE) and Mean Squared Error (MSE):
XGBoost MAE: 0.142 (very good fit).

Linear Regression MAE: 0.000 (indicating potential issues with evaluation or overfitting).

**Forecasting**

Future predictions are made using the trained XGBoost model for the next 24 hours. The results are stored in a DataFrame and printed.

**Conclusion**

The XGBoost model demonstrates promising results for forecasting electrical current based on meteorological data. The feature engineering process significantly improved prediction accuracy.
Future Work
Explore additional models such as LSTM for time series forecasting.
Implement hyperparameter tuning for Linear Regression.
Investigate the impact of seasonal variations on current forecasting.
