import numpy as np
import pandas as pd
import mlflow

dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Fitting Random Forest Regression to the dataset
n_estimators_ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
random_state_ = int(sys.argv[2]) if len(sys.argv) > 2 else 1
from sklearn.ensemble import RandomForestRegressor
with mlflow.start_run():
        regressor = RandomForestRegressor(n_estimators = n_estimators_, random_state = random_state_)
        regressor.fit(X, y)
        mlflow.log_param("n_estimators_", n_estimators_)
        mlflow.log_param("random_state_", random_state_)
        mlflow.sklearn.log_model(regressor, "model")
