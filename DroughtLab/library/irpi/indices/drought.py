"""
Created on Tue Nov 06 09:36:44 2018

@author: c.massari
"""

import numpy as np
from scipy.stats import norm, gamma

def ssi(df_SM, acc_per=1, df_var='sm'):

    # Group data by desired accumulation period and interpolate
    month_values = df_SM[df_var].resample('M').mean()
    month_values = month_values.interpolate()
    accum_period = month_values.rolling(acc_per).mean()

    SSI = accum_period.copy()
    mesi = np.arange(1, 13, 1)

    for jj in mesi:
    
        print(jj)

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

def ssi_gamma(df_SM, acc_per,df_var='sm'):

    # Group data by desired accumulation period and interpolate
    month_values=df_SM[df_var].resample('M').mean()
    month_values=month_values.interpolate()
    accum_period=month_values.rolling(acc_per).mean()

    SSI_gamma = accum_period.copy()

    mesi=np.arange(1,13,1)
    #npixel=np.arange(0,len(SSI.columns))    

    for jj in mesi:
        dfM = np.where(accum_period.index.month == jj)
        series=accum_period.values[dfM]
        wh=~np.isnan(series)
        series1 = series[~np.isnan(series)]
        bp=np.float32((np.sum(series1==0))+1)/(2*(len(series1)+1));
        series2 = series1[np.nonzero(series1)]
        alpha,loc,beta =gamma.fit(series2,floc=0)
        val=gamma.cdf(series1,alpha,loc,beta)
              
                                
        for ii in range(len(series1)):
                if series1[ii]==0:
                    val[ii]=bp;
    
                # Plotting position formula Gringorten 
        sta_inv=norm.ppf(val)
        series[wh]=sta_inv
        SSI_gamma.iloc[accum_period.index.month == jj]=series
    
    return SSI_gamma


def spi_gamma(df_SM, acc_per,df_var='tp'):

    # Group data by desired accumulation period and interpolate
    month_values=df_SM[df_var].resample('M').mean()
    month_values=month_values.interpolate()
    accum_period=month_values.rolling(acc_per).mean()

    SPI_gamma = accum_period.copy()

    mesi=np.arange(1,13,1)
    #npixel=np.arange(0,len(SSI.columns))    

    for jj in mesi:
        dfM = np.where(accum_period.index.month == jj)
        series=accum_period.values[dfM]
        wh=~np.isnan(series)
        series1 = series[~np.isnan(series)]
        bp=np.float32((np.sum(series1==0))+1)/(2*(len(series1)+1));
        series2 = series1[np.nonzero(series1)]
        alpha,loc,beta =gamma.fit(series2,floc=0)
        val=gamma.cdf(series1,alpha,loc,beta)
              
                                
        for ii in range(len(series1)):
                if series1[ii]==0:
                    val[ii]=bp;
    
                # Plotting position formula Gringorten 
        sta_inv=norm.ppf(val)
        series[wh]=sta_inv
        SPI_gamma.iloc[accum_period.index.month == jj]=series
    
    return SPI_gamma

