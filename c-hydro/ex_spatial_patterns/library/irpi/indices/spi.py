# -*- coding: utf-8 -*-
"""
Created on Tue Nov 06 09:36:44 2018

@author: c.massari
"""
import numpy as np

from scipy.stats import norm

def SPIcal(df_PP,acc_per):

    # Group data by desired accumulation period and interpolate
    month_values=df_PP.resample('M').sum()
    month_values=month_values.interpolate()
    accum_period=month_values.rolling(acc_per).mean()

    SPI = accum_period.copy()

    mesi=np.arange(1,13,1)
    #npixel=np.arange(0,len(SSI.columns))
    npixel=np.arange(0,len(SPI.columns))
    for kk in npixel:

            for jj in mesi:
                dfM = accum_period[accum_period.index.month == jj]
    
                series=dfM.values[:,kk]
                series = series[~np.isnan(series)]
                n=len(series)
                bp=np.zeros(len(series))
                
                for ii in range(len(series)):
                        bp[ii]=np.sum(series<=series[ii])
                
                # Plotting position formula Gringorten 
                y=(bp-0.44)/(n+0.12);
                z=norm.ppf(y)
                SPI.iloc[accum_period.index.month == jj,kk]=z
    
    return SPI

"""
mat = scipy.io.loadmat('PP_Esp.mat')
rain=mat['PP_Esp']
rain[rain<0]='0';
rain_m=np.reshape(rain,(4018,7))
date= pd.date_range(start='1/1/2007', end='31/12/2017', freq='D')
df_PP=pd.DataFrame(rain_m,index=date)
    # df_SM must be a pandas dataframe where the index is time and the columns are the pixel contained 
    # in the sh#df_SM1=....
#df_SM1=df1 = df_SM.iloc[:,0:2]

acc_per=1
SPIad=SPIcal(df_PP,acc_per)
df=SPIad.iloc[:,1]
df.plot(x='Date', y='Result')
"""
