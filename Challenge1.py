# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 8:42:27 2020
Daniel J Heathcote Answer to Challenge Question 1 for the Data Incubator

Analysis of CBP Data for Newark Liberty International Airport

@author: danie
"""
#Import necessary toolboxes
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

filenames = os.listdir('Challenge1Data')
df = []

for i in range(len(filenames)):
    filename_curr = 'Challenge1Data/' + filenames[i]
    print('Loading...', filename_curr)
    data = pd.read_excel(filename_curr, header = 3, parse_dates = False)
    df.append(data)
df = pd.concat(df)    
#Sort out column names, times and datatypes before EDA
#Regex statement for time at start of hour, use this for hour by hour binning later on
StartStr = '(\d{4})'
df.columns = ['Airport', 'Terminal', 'Date', 'Interval', 'US-Avg-Wait', 'US-Max-Wait', 'Non-US-Avg-Wait', 'Non-US-Max-Wait', 'Avg-Wait', 'Max-Wait', 'Pax-0-15', 'Pax-16-30', 'Pax-31-45', 'Pax-46-60', 'Pax-61-90', 'Pax-91-120', 'Pax-120+', 'Excluded', 'Total', 'Flights', 'Booths']
df['StartTime'] = df['Interval'].str.extract(StartStr).astype('int')/100   


#Set up full date string, this takes a little while to process. There should be a quicker and easier way of processing this
for i2 in range(len(df)):
    print("Percent Complete: %3.2f " % (i2/len(df)*100))
    df['Date'].iloc[i2] = df['Date'].iloc[i2] + timedelta(hours=df['StartTime'].iloc[i2])
    
# Calculate time spans to group by, Monthly mean figures will be the standard
    
df['Year']      = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Week']      = df['Date'].dt.week
df['Month']     = df['Date'].dt.month
df['Hour']      = df['Date'].dt.hour
df['DayOfYear']  = df['Date'].dt.dayofyear

# Additional variables
df['PaxPerFlight'] =    df['Total']/df['Flights']
df['BoothsPerFlight'] = df['Booths']/df['Flights']
df['PaxPerBooth'] = df['Total']/df['Flights']

# Subset for Terminal B and C
# Group by day of year to get average per day and totals per day and sum
# then  group by Month to get total monthly and daily figures
#Calculate Sum 
TermB_2018 = df[(df['Year']== 2018) & (df['Terminal'] == 'Terminal B')].groupby(['DayOfYear']).sum()
TermC_2018 = df[(df['Year']== 2018) & (df['Terminal'] == 'Terminal C')].groupby(['DayOfYear']).sum()
Data_2020 = df[(df['Year']== 2020)].groupby(['DayOfYear']).sum()
Data_2019 = df[(df['Year']== 2019)].groupby(['DayOfYear']).sum()

#Define columns where mean, std make more sense than sum
listformean = ['Month',  'US-Avg-Wait', 'US-Max-Wait', 'Non-US-Avg-Wait', 'Non-US-Max-Wait', 'Avg-Wait', 'Max-Wait', 'PaxPerFlight', 'PaxPerBooth', 'BoothsPerFlight']

TermB_2018[listformean] = df[(df['Year']== 2018) & (df['Terminal'] == 'Terminal B')].groupby(['DayOfYear']).mean()[listformean]
TermC_2018[listformean] = df[(df['Year']== 2018) & (df['Terminal'] == 'Terminal C')].groupby(['DayOfYear']).mean()[listformean]

#Calculate Information for 2020, looking at the effect of the coronavirus on total flights and passengers per day
Data_2020[listformean] = df[(df['Year']== 2020)].groupby(['DayOfYear']).mean()[listformean]
Data_2019[listformean] = df[(df['Year']== 2019)].groupby(['DayOfYear']).mean()[listformean]

#Exclude month from the second set of averages, its now the index
listformean.append('Flights')
listformean2 = listformean[1:] 

TermB_2018_Avg = TermB_2018.groupby(['Month']).sum()
TermB_2018_Avg[listformean2] = TermB_2018.groupby(['Month']).mean()[listformean2]
TermB_2018_Std = TermB_2018.groupby(['Month']).sum()
TermB_2018_Std[listformean2] = TermB_2018.groupby(['Month']).std()[listformean2]

TermC_2018_Avg = TermC_2018.groupby(['Month']).sum()
TermC_2018_Avg[listformean2] = TermC_2018.groupby(['Month']).mean()[listformean2]
TermC_2018_Std = TermC_2018.groupby(['Month']).sum()
TermC_2018_Std[listformean2] = TermC_2018.groupby(['Month']).std()[listformean2]


# Monthly information for terminals B and C for 2018

fig= plt.figure(figsize = [20,10] )
ax = fig.add_axes([0,0, 0.25, 0.25]) # main axes
ax.errorbar(TermB_2018_Avg.index, TermB_2018_Avg['Avg-Wait'], TermB_2018_Std['Avg-Wait'], label = 'Terminal B', linestyle = '-', marker = '.')
ax.errorbar(TermC_2018_Avg.index, TermC_2018_Avg['Avg-Wait'], TermC_2018_Std['Avg-Wait'], label = 'Terminal C', linestyle = '-', marker = '.')
ax.set(xticks = np.arange(1,13))
ax.set(ylim = (0,50))
ax.set(xticklabels = ['Jan','Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set(xlabel = 'Month in 2018')
ax.set(ylabel = 'Average Passenger Wait Time')
ax.legend()
plt.show()

# Graph indicates that the mean for Terminal C wait times is generally lower, but there are some instances where the stdev of the mean value is higher

MonthlyAvg = df.groupby(['Year','Month']).mean()
MonthlyStd = df.groupby(['Year', 'Month']).std()

def GenerateDates(df, year, month, day):
    """

    Parameters
    ----------
    df : Dataframe
        Dataframe with days specified as 'days of year' .
    year : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = df.reset_index()
    a = []
    for i in range(len(df)):
        s = datetime(year, month, day) + timedelta(days=int(df['DayOfYear'].iloc[i]))
        a.append(s)
        
    df.set_index('DayOfYear')
    df['Date'] = a
    return df

def MovingAv(df, varname,n):
    
    a = pd.Series(df[varname]).rolling(window=n).mean().iloc[n-1:].values
    a_len = len(a)
    df_len = len(df)
    diff_len = df_len-a_len
    for i in range(diff_len):
        a = np.append(a, np.nan)
    
    df[varname+ '_MovAvg'] = a 
    return df

# Generate the date array for plotting
Data_2020 = GenerateDates(Data_2020, 2020, 1, 1)


Data_2019 = GenerateDates(Data_2019, 2020, 1, 1)

#Moving average filter the data to allow for better presentation and to remove
# weekly fluctuations

Data_2020 = MovingAv(Data_2020, 'Total', 7)
Data_2019 = MovingAv(Data_2019, 'Total', 7)

Data_2020.set_index('Date')
Data_2019.set_index('Date')

mean_2019 = np.mean(Data_2019['Total'])

# Fit linear model for data from June 15th onwards
# Programmatically finding the date where the passenger numbers start to increase
#Generate fit and forward prediction to find intercept with historical mean data
Data_2020_tofit = Data_2020[Data_2020['Date'] >= datetime(2020,6,15)]
fit = np.polyfit(Data_2020_tofit['DayOfYear'],Data_2020_tofit['Total'],1)
theoretical_fit = pd.DataFrame({'DayOfYear':np.arange(np.min(Data_2020_tofit['DayOfYear']),1000,1)})
theoretical_fit['Total'] = theoretical_fit['DayOfYear']*fit[0]+fit[1] 
theoretical_fit = GenerateDates(theoretical_fit, 2020,1, 1)
test_stat = theoretical_fit['Total'] > mean_2019
return2019 = theoretical_fit['DayOfYear'].iloc[test_stat[test_stat == True].index[0]]
level_2019 = [datetime(2020, 1, 1) + timedelta(days=int(return2019)), mean_2019]
                   
# Plot the results
fig2 = plt.figure(figsize = [30,15] )
ax2 = fig2.add_axes([0,0, 0.25, 0.25]) # main axes
ax2.plot(Data_2020['Date'], Data_2020['Total'], label = '2020', linestyle = 'none', marker = '.', color = 'b', alpha = 0.2)
ax2.plot(Data_2020['Date'], Data_2020['Total_MovAvg'], linestyle = '-', marker = None, color = 'b', lw = 2)
ax2.plot(Data_2019['Date'], Data_2019['Total'], label = '2019 (Shifted by 12 months)', linestyle = 'none', marker = '.', color = 'g', alpha = 0.2)
ax2.plot(Data_2019['Date'], Data_2019['Total_MovAvg'], linestyle = '-', marker = None, color = 'g', lw = 2)
ax2.plot([Data_2019['Date'].iloc[0],Data_2019['Date'].iloc[-1]], [mean_2019, mean_2019], linestyle = '--', lw = 2, color = 'k')

# Add points of interest
# Trump travel executive order issued
ax2.plot([datetime(2020, 3, 11, 0, 0),datetime(2020, 3, 11, 0, 0)], [10000, 20000], linestyle = '--', lw = 2, color = 'gray')

# Location of minimum passengers 
ax2.plot(datetime(2020, 4, 14, 0, 0), [51], linestyle = 'none', marker = 'x',lw = 3, color = 'k', markersize = 10)

#Add annotations
ax2.annotate(xy=[datetime(2020, 4, 14, 0, 0),100], s = '14th April: \nMinimum Pax = 51', xytext = (datetime(2020,4,14),5000), arrowprops = dict(arrowstyle = '->'))
ax2.annotate(xy=[datetime(2020, 3, 11, 0, 0),20000], s = '11th March: Trump Excutive Order \n Restricting International Flights', xytext = (datetime(2020,3,14),23000), arrowprops = dict(arrowstyle = '->'))
ax2.annotate(xy=[datetime(2020,11, 15, 0, 0),mean_2019], s = ('2019 Average Pax \n per Day: %3i' % mean_2019), xytext = (datetime(2020,10,15),23000), arrowprops = dict(arrowstyle = '->'))
ax2.set(ylim = (-1000,26000))
ax2.set(xlim = (datetime(2020,1,1),datetime(2021,1,1)))
ax.set(xticks = np.arange(1,13))
#ax.set(xticklabels = ['Jan','Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax2.set(xlabel = 'Date (Year-Month)')
ax2.set(ylabel = 'Total Daily Passengers')
ax2.legend(loc = 4)

plt.show()

 

fig3 = plt.figure(figsize = [30,15])
ax3 = fig3.add_axes([0,0, 0.25, 0.25])

ax3.plot(Data_2020_tofit['Date'], Data_2020_tofit['DayOfYear']*fit[0]+fit[1], color = 'b', label = 'fit')
ax3.plot(Data_2020_tofit['Date'], Data_2020_tofit['Total'], linestyle = 'none', marker = '.', color = 'b', label = 'Raw Data')
ax3.set(xlabel = 'Date')
ax3.set(ylabel = 'Total Daily Passengers')
ax3.legend()
plt.show()

#Calculate return to mean 2019 levels, using linear regression model

# Plot the linear regression
fig4 = plt.figure(figsize = [30,15] )
ax4 = fig4.add_axes([0,0, 0.25, 0.25]) # main axes
ax4.plot(Data_2020['Date'], Data_2020['Total'], label = '2020 ', linestyle = 'none', marker = '.', color = 'b', alpha = 0.2)
ax4.plot(Data_2020['Date'], Data_2020['Total_MovAvg'], linestyle = '-', marker = None, color = 'b', lw = 2)
ax4.plot(Data_2019['Date'], Data_2019['Total'], label = '2019 (Shifted by 12 months)', linestyle = 'none', marker = '.', color = 'g', alpha = 0.2)
ax4.plot(Data_2019['Date'], Data_2019['Total_MovAvg'], linestyle = '-', marker = None, color = 'g', lw = 2)
ax4.plot([Data_2019['Date'].iloc[0],datetime(2023,1,1)], [mean_2019, mean_2019], linestyle = '--', lw = 2, color = 'k')

# Add Annotations

ax4.annotate(xy=[level_2019[0], level_2019[1]], s = '13th March 2021: \nPredicted return to \n2019 passenger levels', xytext = (datetime(2022,5,1),10000), arrowprops = dict(arrowstyle = '->'))
ax4.annotate(xy=[datetime(2021,11, 15, 0, 0),mean_2019], s = ('2019 Average Pax \n per Day: %3i' % mean_2019), xytext = (datetime(2021,10,15),23000), arrowprops = dict(arrowstyle = '->'))

# Predicted Return to pre-coronavirus levels

ax4.plot(theoretical_fit['Date'], theoretical_fit['Total'], linestyle = '--', marker = None, color = 'gray', lw = 2)
ax4.set(xlabel = 'Date (Year - Month)')
ax4.set(ylabel = 'Total Daily Passengers')
ax4.legend(loc = 4)
ax4.set(ylim = (-1000,26000))
ax4.set(xlim = (datetime(2020,1,1),datetime(2023,1,1)))
plt.show()


    



