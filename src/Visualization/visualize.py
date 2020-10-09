#Read in libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Plot:
    def __init__(self, figure_size = None, color = None, dataframe = None):
        self.figure_size = figure_size
        self.color = color
        self.dataframe = dataframe
    
    def heat_map(self, figure_size=None):
        """
        Function takes a dataframe and returns a heatmap reflecting the correlations between 
        each of the features of said dataframe
        """
        f, ax = plt.subplots(figsize = self.figure_size)
        corr = self.dataframe.corr()
        j = sns.heatmap(corr, ax = ax)
        return j

    #def feature_histogram(df,figure_size = None )

