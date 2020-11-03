#Read in libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

class EDA_Viz:
    """
    Class generates an object with attributes that allow you to generate one of the following visualizations:
        - Heat Map
        - Histrogram
        - Boxplot
        - Scatterplot
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe      
    
    def heat_map(self, ax=None):
        """Method generates heatmap.

        Args:
            ax (matplotlib.axes, optional): Plot visualization on an externally created plot. Defaults to None.

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
    
    def histogram(self,feature, ax = None,bins=30):
        """Method generates histogram

        Args:
            feature (series): feature within your data you would like to build a histogram out of
            ax (matplotlib.axes, optional): Axis you would like to place visualization on. Defaults to None.
            bins (int, optional): Bins used for histogram. Defaults to 30.

        Returns:
            j : histogram graphic
        """
        j = sns.histplot(self.dataframe, x=feature,
                        color= "red",alpha = .5, bins = bins,
                        kde= True, ax=ax)
        return j

    def boxplot(self,cont_var,
                 cat_var= None, hue = None, 
                order = None, showfliers = True,
                ax = None):
        """Method generates Boxplot

        Args:
            cont_var (panda series): continuos variable you would like to be modeled in a boxplot visualization
            cat_var (panda series, optional): categorical variable you would like to group the cont_var into for the visualization. 
            Defaults to None.
            hue (pandas series, optional): color grouping you would like to see in the visualization. Defaults to None.
            order (list, optional): Order you would like to see the categorical groupings to be placed in visualization. Defaults to None.
            showfliers (bool, optional): Select whether or not to include outliers in your visualization. Defaults to True.
            ax (matplotlib.axes, optional): Axis you would like to place visualization on. Defaults to None.

        Returns:
            matplotlib visualiztion: Boxplot graphic
        """
            
        j = sns.boxplot(x = cat_var, y = cont_var, 
                        hue = hue, order=order,
                        data=self.dataframe, showfliers=showfliers,
                        ax = ax)
        return j

    
    def scatterplot(self, x, y,
                    hue=None, style = None,
                    sizes = None, ax = None):
        """Method generates scatterplot

        Args:
            y (pandas Series): continuous variable you would like to add to scatter plot on y-axis. Defaults to None.
            x (pandas Series): continuous variable you would like to add to scatter plot on x-axis. Defaults to None.
            hue (pandas Series, optional): Grouping variable that will produce points with different colors.. Defaults to None.
            style (pandas Series, optional): Grouping variable that will produce points with different markers. Defaults to None.
            sizes (pandas Series, optional): An object that determines how sizes are chosen when size is used. Defaults to None.
            ax (matplotlib.axes, optional): Axis you would like to place visualization on. Defaults to None.

        Returns:
            matplotlib visualiztion: scatterplot graphic
        """

        j = sns.scatterplot(x=x, y=y, 
                            hue=hue, style = style,
                            sizes = sizes, data = self.dataframe, 
                            ax = ax)
        return j    


    def qqplot(self,feature, fit= True, line = '45',
                ax=None):
        import scipy.stats as stats
        j= sm.qqplot(self.dataframe[feature],stats.t, fit= fit,
                line=line, ax=ax)
        return j

    def pairplot(self, corner = True, plot_kws= None,
                diag_kws= None,height=2.5, aspect=1):
        
        j= sns.pairplot(self.dataframe, corner=corner,plot_kws= plot_kws,
                        diag_kws= diag_kws, height=height, aspect=aspect)
        return j


def set_aesthetics(title = '', xlabel = '',ylabel = '', 
                    fontdict =None , ax= None):
    """Set plot aesthetics

     Args:
        title (str, optional): String you would like to use for the title. Defaults to ''.
        xlabel (str, optional): String you would like to use for the xlabel. Defaults to ''.
        ylabel (str, optional): String you would like to use for the ylabel. Defaults to ''.
        fontdict (dictionary, optional): Dictionary that contains fontsizes "title_fontsize" and "label_fontsize". Defaults to None.
        ax (matplotlib.axes, optional): Axis you would like to place visualization on. Defaults to None.
    """
    ax.set_title(title, fontsize = fontdict['title_fontsize'])
    ax.set_xlabel(xlabel, fontsize = fontdict['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize = fontdict['label_fontsize'])

    return ax
