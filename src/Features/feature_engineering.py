#Read in libraries
import numpy as np
import pandas as pd

#Import the statsmodels goodies
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
    

class Tests:
    def __init__(self, dataframe):
        self.data = dataframe
        
    def variance_inflation_score(self):
        """Returns VIF scores for a Dataframe

        Returns:
            Pandas Series: Returns VIF scores for each feature withing a dataframe
        """
        
        X= add_constant(self.data)
        X= X.select_dtypes(include = ['int64', 'float64'])
        VIF= pd.Series([variance_inflation_factor(X.values, i) 
                        for i in range(X.shape[1])],index= X.columns).sort_values(ascending = False)
        return VIF
#Encode categorical variables using column transformer

#Scale/Normalize variables

#Test for multicollinearity 
#Test for normality