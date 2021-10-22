import numpy as np
import pandas as pd
import mlflow

dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')
X = dataset['Temperature'].values
y = dataset['Revenue'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run():
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X_train(-1,1), y_train.reshape(-1,1))
        mlflow.sklearn.log_model(regressor, "model")
