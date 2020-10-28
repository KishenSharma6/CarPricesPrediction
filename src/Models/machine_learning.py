#Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

class Linear_Regression:
    def __init__(self, dataframe):
        self.data = dataframe


    def pipeline(self ):
        pass


def regression_metrics(predictions, actual):
    errors = predictions - actual
    mape = np.mean(np.abs(errors)/np.abs(actual)) * 100
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    r2 = r2_score(actual, predictions)
    metrics = {'MAE': mae,'MAPE':mape,'MSE': mse, 'RMSE':rmse}
    return(metrics)
