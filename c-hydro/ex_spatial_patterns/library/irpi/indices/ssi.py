# -*- coding: utf-8 -*-
"""
Created on Tue Nov 06 09:36:44 2018

@author: c.massari
"""
import numpy as np
from scipy.stats import norm

def SSIcal(df_SM, acc_per):

    # Group data by desired accumulation period and interpolate
    month_values=df_SM.resample('M').mean()
    month_values=month_values.interpolate()
    accum_period=month_values.rolling(acc_per).mean()

    SSI = accum_period.copy()

    mesi=np.arange(1,13,1)
    #npixel=np.arange(0,len(SSI.columns))
    npixel=np.arange(0,len(SSI.columns))
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
                SSI.iloc[accum_period.index.month == jj,kk]=z
    
    return SSI

"""
mat = scipy.io.loadmat('SM_Esp.mat')
sm=mat['SM']
sm_m=np.reshape(sm,(5844,1141))
sm_m=sm_m/100;

date= pd.date_range(start='1/1/2007', end='31-Dec-2014 12:00:00 ', freq='12H')

df_SM=pd.DataFrame(sm_m,index=date)
    # df_SM must be a pandas dataframe where the index is time and the columns are the pixel contained 
    # in the shp file numbered from 1 to N.

#df_SM1=df1 = df_SM.iloc[:,0:2]
acc_per=1
SSIad=SSIcal(df_SM,acc_per)
df=SSIad.iloc[:,400]
df.plot(x='Index', y='Result')
"""