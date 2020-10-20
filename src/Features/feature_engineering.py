#Read in libraries
import numpy as np
import pandas as pd

#Import all the sklearn goodies
from sklearn.preprocessing import OneHotEncoder

#Import the statsmodels goodies
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

class Preprocessor:
    def __init__(self, dataframe):
        self.data = dataframe

#Encode categorical variables using column transformer

    def nominal_encoder(self, nominal_features, drop= None,
                        categories = 'auto'):
        encoder= OneHotEncoder(drop = drop, handle_unknown='ignore',
                                categories= categories)
        x = encoder.fit_transform(self.data[nominal_features])
        return x
        

class Tests:
    

    def __init__(self, dataframe):
        self.data = dataframe
        
    def variance_inflation_score(self):
        
        X= data.add_constant()
#Encode categorical variables using column transformer

#Scale/Normalize variables

#Test for multicollinearity 
#Test for normality