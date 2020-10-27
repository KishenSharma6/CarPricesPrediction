#Read in standard libraries
import numpy as np
import pandas as pd

#Import the statsmodels goodies
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#Import scipy goodies
from scipy import stats
    

class Tests:
    def __init__(self, dataframe):
        self.data = dataframe
        
        """Class consists of different statistical tests to better understand 
        data for machine learning purposes
        """
    def variance_inflation_score(self):
        
        X= add_constant(self.data)
        X= X.select_dtypes(include = ['int64', 'float64'])
        VIF= pd.Series([variance_inflation_factor(X.values, i) 
                        for i in range(X.shape[1])],index= X.columns).sort_values(ascending = False)
        return VIF

    def skew_measurement(self):
        return self.data.skew()

    def shapiro_wilks(self,target):
        shapiro_test = stats.shapiro(self.data[target])
        print('Shapiro statistic for %s: %s\nP-Value: %s\n' % (target, shapiro_test.statistic, shapiro_test.pvalue))


class Transformations(Tests):

    """Class consists of methods for different transformations intended to be 
    applied to a target feature in a dataframe for machine learning.
    """
    def __init__(self, data):
        super().__init__(data)

    def square_root(self, target):
        return np.sqrt(self.data[target])

    def cube_root(self, target):
        return np.cbrt(self.data[target])

    def log_transformation(self, target):
        return np.log(self.data[target])
    
    def boxcox_transformation(self, target):
        return stats.boxcox(self.data[target])


