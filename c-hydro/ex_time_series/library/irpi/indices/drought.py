"""
Created on Tue Nov 06 09:36:44 2018

@author: c.massari
"""

import numpy as np
from scipy.stats import norm

def ssi(df_SM, acc_per=1, df_var='sm'):

    # Group data by desired accumulation period and interpolate
    month_values = df_SM[df_var].resample('M').mean()
    month_values = month_values.interpolate()
    accum_period = month_values.rolling(acc_per).mean()

    SSI = accum_period.copy()
    mesi = np.arange(1, 13, 1)

    for jj in mesi:

        dfM = accum_period[accum_period.index.month == jj]

        series = dfM.values
        series = series[~np.isnan(series)]
        n = len(series)
        bp = np.zeros(len(series))

        for ii in range(len(series)):
            bp[ii] = np.sum(series <= series[ii])

        # Plotting position formula Gringorten
        y = (bp - 0.44) / (n + 0.12);
        z = norm.ppf(y)
        SSI.iloc[accum_period.index.month == jj] = z

    return SSI

def spi(df_PP, acc_per=1, df_var='tp'):

    # Group data by desired accumulation period and interpolate
    month_values = df_PP[df_var].resample('M').sum()
    month_values = month_values.interpolate()
    accum_period = month_values.rolling(acc_per).mean()

    SPI = accum_period.copy()
    mesi = np.arange(1, 13, 1)

    for jj in mesi:
        dfM = accum_period[accum_period.index.month == jj]

        series = dfM.values
        series = series[~np.isnan(series)]
        n = len(series)
        bp = np.zeros(len(series))

        for ii in range(len(series)):
            bp[ii] = np.sum(series <= series[ii])

        # Plotting position formula Gringorten
        y = (bp - 0.44) / (n + 0.12);
        z = norm.ppf(y)
        SPI.iloc[accum_period.index.month == jj] = z

    return SPI
