import numpy as np
import pandas as pd
import mlflow

dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
with mlflow.start_run():
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X(-1,1), y.reshape(-1,1))
        mlflow.sklearn.log_model(regressor, "model")
