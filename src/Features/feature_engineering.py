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

    def skew_measurement(self):
        return self.data.skew()

    def shapiro_wilks(self,target):
        shapiro_test = stats.shapiro(self.data[target])
        print('Shapiro statistic: %s \nP-Value: %s' % (shapiro_test.statistic, shapiro_test.pvalue))


class Transformations(Tests):
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


#Encode categorical variables using column transformer

#Scale/Normalize variables

#Test for multicollinearity 
#Test for normality