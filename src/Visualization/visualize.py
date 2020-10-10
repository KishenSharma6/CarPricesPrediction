#Read in libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDA_Viz:
    def __init__(self, dataframe = None, color = None):
        self.color = color
        self.dataframe = dataframe
    
    def heat_map(self, ax=None):
        """
        Function takes a dataframe and returns a heatmap reflecting the correlations between 
        each of the features of said dataframe
        """
        #Create correlation matrix
        corr = self.dataframe.corr()

        #Create mask 
        mask = np.triu(np.ones_like(corr, dtype=np.bool))

        #Minor adjustments to inclde edge values in mask of heat map
        mask = mask[1:,:-1]
        corr = corr.iloc[1:,:-1].copy()

        #Create heatmap plot using correlation matrix
        j = sns.heatmap(mask=mask, cmap='Blues', 
                        linewidths=.25, annot=True, 
                        fmt=".2f", cbar_kws={"shrink": .8},
                        data=corr, ax = ax)
        return j
    
    def histogram(self,feature = None, ax = None,bins=30):
        j = sns.histplot(self.dataframe, x=feature,
                        color= self.color, bins = bins,
                        kde= True, ax=ax)
        return j

    def boxplot(self, cat_var= None, cont_var=None, 
                ax = None):
        j = sns.boxplot()

    
#    #def scatterplot(self):
            
        
