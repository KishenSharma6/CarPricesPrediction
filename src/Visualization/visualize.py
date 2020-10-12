#Read in libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDA_Viz:
    """
    Class generates an object with attributes that allow you to generate one of the following visualizations:
        - Heat Map
        - Histrogram
        - Boxplot
        - Scatterplot
    """
    def __init__(self, dataframe = None, color = None):
        self.color = color
        self.dataframe = dataframe
    
    def heat_map(self, ax=None):
        """Generates heatmap.

        Args:
            ax (optional): Plot visualization on an externally created plot. Defaults to None.

        Returns:
            Heatmap visualization
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
        """Generates histogram

        Args:
            feature (data, optional): [description]. Defaults to None.
            ax ([type], optional): [description]. Defaults to None.
            bins (int, optional): [description]. Defaults to 30.

        Returns:
            [type]: [description]
        """
        j = sns.histplot(self.dataframe, x=feature,
                        color= self.color, bins = bins,
                        kde= True, ax=ax)
        return j

    def boxplot(self, cat_var= None, 
                cont_var=None, hue = None, 
                order = None, showfliers = True,
                ax = None):
        j = sns.boxplot(x = cat_var, y = cont_var, 
                        hue = hue, order=order,
                        data=self.dataframe, showfliers=showfliers,
                        ax = ax)
        return j

    
    def scatterplot(self, x = None, y = None,
                    hue=None, style = None,
                    sizes = None, ax = None):
        j = sns.scatterplot(x=x, y=y, 
                            hue=hue, style = style,
                            sizes = sizes, data = self.dataframe, 
                            ax = ax)
        return j
            
        
