# -*- coding: utf-8 -*-

"""Script to 

Created on Thu Mar 23 14:25:24 2023

Includes information on 

Parameters
----------

@author: nquer
@date: 2023-03-23
@license = MIT -- https://opensource.org/licenses/MIT

"""

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

#%% Specify Parameters

#Name of input file with extension
infile_names = ['mt washington.csv']
#Data is in US Customary Units

#Create start and end dates
start_date = '1948-01-01'
end_date = '2022-12-31'

#%% Load Data

#Load data and set index
data = pd.read_csv(infile_names[0], comment='#',
                   parse_dates=['DATE'], index_col=['DATE'])

#Drop unnecessary columns
columns=['SNOW','TMAX','TMIN']
data = data[columns]

#Cutting data to start and end dates
data=data[start_date:end_date]

#Drop NaN values from the dataset before analysis
data = data.dropna()

#Creating an average temperature column using min and max temperatures from each day
data['TAVG']=(data.TMIN+data.TMAX)/2

#%% Function to plot data

def plotdata(data,title,ylabel):
    plt.style.use('seaborn-whitegrid') # Use a custom style for the plot
    fig, ax = plt.subplots() # Create plot to display snowfall data
    ax.plot(data.index, data, color='#e74c3c', linewidth=1, label='Snowfall') # Plot snowfall data with a red line and thicker linewidth
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold') # Add y-axis label with a larger font size and bold text
    ax.set_title(title, fontsize=16, fontweight='bold') # Add plot title with a larger font size and bold text
    plt.xticks(fontsize=12, fontweight='bold') # Increase the font size and weight of the x-tick labels
    ax.grid(axis='both', color='#c7c7c7', linestyle='-', linewidth=1) # Show the grid lines on both axes
    plt.show() # Show the plot

#%% Create dataframe consisiting of just winter season, dec1-mar31

#Trim data to first Dec1 and last Mar 30
startdate=data.loc[(data.index.month==12)].index[0]
enddate=data.loc[(data.index.month==3)].index[-1]
winterdata = data[startdate:enddate].copy()

#Numbering each winter
winterdata['month']=winterdata.index.month
winterdata['winteryear']=winterdata.index.year
winterdata.loc[winterdata['month']<=3,'winteryear']=winterdata.loc[winterdata['month']<=3,'winteryear'] -1

#Replace data with Nan outside winter months (dec,jan,feb,mar)
winterdata.loc[~((winterdata.index.month >= 12) | (winterdata.index.month <= 3))] = pd.NA

#%% Winter Year Calculations

#Create new dataframe with yearly values for snowfall sum and average temperature
data_annual = winterdata[['SNOW']].groupby(winterdata.winteryear).sum()
data_annual['tavg'] = winterdata['TAVG'].groupby(winterdata.winteryear).mean()

#Rename columns in data_annual
data_annual.rename(columns = {"SNOW":"snowfall (in)",'tavg':'Temp (F)'},inplace=True)

#%% Plot data

#Plot raw snowfall data
plotdata(data['SNOW'],'Mt. Washington, NH Snowfall', 'Snowfall (in)')

#Plot raw temperature data
plotdata(data['TAVG'],'Mt. Washington, NH Temperature', 'Temperature (F)')

#Plot average winter temperature
plotdata(data_annual['Temp (F)'],'Mt. Washington, NH Average Winter Temperature','Temperature (F)')

#Plot total winter snow depths
plotdata(data_annual['snowfall (in)'], 'Mt. Washington, NH Total Winter Snowfall','Snow Depth (in)')

#%% Perform statistics to determine signifcant change in snowfall

#Perform nonlinear statistics to determine significance

def lsqplot(x, y, title, nonparam=False, xlabel=None, ylabel=None, xtoplot=None):
    """ Plot least-squares fit to data.

    Function to create scatter plot of x-y data, including regression
    line(s) and annotation with regression parameters and statistics.

    Parameters
    ----------
    x : array or series
        Independent 1-D numerical data
    y : array or series
        Independent 1-D numerical data with the same length as x    
    nonparam : boolean
        Toggle to determine whether to include non-parametric fit 
    xlabel : string
        Text to use as x-label on plot
    ylabel : string
        Text to use as y-label on plot
    xtoplot : array or series
        Alternative positions for x data to use in plotting not fitting;
        shuold be 1-D numerical or datetime data with same length as x
    """

    sen_coeff = stats.theilslopes(y, x, 0.95)
    tau = stats.kendalltau(x, y)

    if xtoplot is None:
        xtoplot = x

    fig, ax = plt.subplots()
    ax.plot(xtoplot, y, 'k.')
    xx = ax.get_xlim()
    yy = ax.get_ylim()
    ax.set_ylim(bottom=yy[0]*.95)

    if nonparam is True:
        ax.plot(xtoplot, sen_coeff.intercept + sen_coeff.slope *
                x, 'r-', label='Theil-Sen regression')
        ax.annotate(f'Theil-Sen slope = {sen_coeff.slope:.4f} +/- {0.5*(sen_coeff.high_slope - sen_coeff.low_slope):.4f}',
                    xy=(xx[1]-0.05*(xx[1]-xx[0]), yy[0] + 0.05*(yy[1]-yy[0])),
                    horizontalalignment='right')
        ax.annotate(f'Tau correlation = {tau.correlation:.3f}; p = {tau.pvalue:.6f}',
                    xy=(xx[1]-0.05*(xx[1]-xx[0]), yy[0] - 0.001*(yy[1]-yy[0])),
                    horizontalalignment='right')

    ax.set_title(title, fontsize=16, fontweight='bold') # Add plot title with a larger font size and bold text
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold') # Add x-axis label with a larger font size and bold text
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold') # Add y-axis label with a larger font size and bold text
    ax.legend(frameon=True, loc='upper right')
    ax.grid(axis='both', color='#c7c7c7', linestyle='-', linewidth=1) # Show the grid lines on both axes
    plt.show()
 
#%% Perform statistics to determine significant change in snowfall

x = data_annual.index.values
y = data_annual['snowfall (in)']
lsq_coeff = stats.linregress(x, y)
slope = lsq_coeff.slope    
intercept = lsq_coeff.intercept
title = 'Winter Snowfall Regressions Statistics'

#Plot the data
lsqplot(x, y, title, nonparam=True, xtoplot = x, ylabel='Snowfall (in)')

#%% Perform statistics to determine signifcant change in temperature

x = data_annual.index.values
y = data_annual['Temp (F)']
lsq_coeff = stats.linregress(x, y)
slope = lsq_coeff.slope    
intercept = lsq_coeff.intercept
title = 'Winter Temperature Regressions Statistics'

#Plot the data
lsqplot(x, y, title, nonparam=True, xtoplot = x, ylabel='Temperature (F)')

#%% Perform statistics to determine signifcant change in snow depth vs temp for each winter season

x = data_annual['Temp (F)']
y = data_annual['snowfall (in)']
lsq_coeff = stats.linregress(x, y)
slope = lsq_coeff.slope    
intercept = lsq_coeff.intercept
title = 'Winter Temperature vs Snowfall Regressions Statistics'

#Plot the data
lsqplot(x, y,title, nonparam=True, xtoplot = x,xlabel='Temperature (F)', ylabel='Snowfall (in)')
