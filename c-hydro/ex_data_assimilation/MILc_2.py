# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:20:11 2016

@author: christian_massari, Luca Brocca
"""

import numpy as np
import scipy as sc
from scipy import misc
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys

#%%
def IUH_comp(gamma,Ab,dt,deltaT):
    """
    %% % -------------------------------------------------------------------------------
    #% Calculation of Geomorphological Instantaneous Unit Hydrograph
    #% ---------------------------------------------------------------------------------
    """
    Lag=(gamma*1.19*Ab**0.33)/deltaT
    hp=0.8/Lag
    data=np.loadtxt('IUH.txt')
    t=data[:,0]*Lag
    IUH_0=data[:,1]*hp
    ti=np.arange(0,max(t),dt)
    IUH=np.interp(ti,t,IUH_0)
    return IUH
#%%

def IUH_NASH(n,gamma,Ab,dt,deltaT):
    
    """
    % -------------------------------------------------------------------------------
    % Calculation of Nash Instantaneous Unit Hydrograph
    % -------------------------------------------------------------------------------
    """
    K=(gamma*1.19*Ab**.33)/deltaT
    time=np.arange(0,100,dt)
    IUH=((time/K)**(n-1)*np.exp(-time/K)/sc.special.factorial(n-1)/K)
    return IUH

#%% 
class Perf:
    
    def __init__(self,pd_ser):
        self.pd_ser=pd_ser
        self.descrition="This class calculates the agreement between two time series thorugh different performance scores"
        self.author="Christian Massari"
    
    def RMSE(self):
        temp1=((self.pd_ser['S']-self.pd_ser['O'])**2)**0.5
        return temp1.mean()
        
    def NS(self):
        temp1=((self.pd_ser['S']-self.pd_ser['O'])**2)
        temp2=(self.pd_ser['O']-self.pd_ser['O'].mean())**2
        NS=1-temp1.sum()/temp2.sum()
        return NS

    def ANSE(self):
        temp1=((self.pd_ser['S']-self.pd_ser['O'])**2)
        temp2=((self.pd_ser['O']-self.pd_ser['O'].mean())**2)
        temp3=(self.pd_ser['O']+self.pd_ser['O'].mean())
        
        temp4=temp3*temp1
        temp5=temp3*temp2
    
        ANSE=1-temp4.sum()/temp5.sum()
        return ANSE

    def R(self):
        return self.pd_ser['O'].corr(self.pd_ser['S'])

        
    def NS_lnQ(self):
        temp1=(np.log(self.pd_ser['S']+0.00001)-np.log(self.pd_ser['O']+0.00001))**2
        temp2=((np.log(self.pd_ser['O']+0.00001)-(np.log(self.pd_ser['O']+0.00001)).mean())**2)
        return 1-temp1.sum()/temp2.sum()
        self.pd_ser['O']['a'].corr(self.pd_ser['O'])

#%% Read data from file


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:26:54 2016

@author: christian_massari
"""

def MILC(name,data_input,PAR,Ab,fig,Wobs=[],K=0):
    

    PIO=data_input['P']
    TEMPER=data_input['T']
    Qobs=data_input['Q']
    N=len(Qobs)
    data_input.index=pd.to_datetime(data_input.index)
    
    
    MESE = data_input.index.month.values
    
    #% READ MODEL PARAMETERS
    
    W_p       = PAR[0] # initial conditions, fraction of W_max (0-1)
    W_max     = PAR[1] # Field capacity
    m2        = PAR[2] # exponent of drainage
    Ks        = PAR[3] # Ks parameter of infiltration and drainage
    Nu        = PAR[4] # fraction of drainage verusu interflow
    gamma1    = PAR[5] # coefficient lag-time relationship
    Kc        = PAR[6] # parameter of potential evapotranspiration
    alpha     = PAR[7] # runoff exponent
    
    
    delta_T = 24      # input data time step in hour
    dt      = 0.2     # computation time step in hour
    Ks      = Ks*24

    
    #%  Potential Evapotranspiration parameter
    L=np.array([0.2100,0.2200,0.2300,0.2800,0.3000,0.3100,0.3000,0.2900,0.2700,0.2500,0.2200,0.2000])
    
    Ka=1.26
    T=TEMPER.values
    EPOT=(T>0)*(Kc*(Ka*L[MESE-1]*(0.46*T+8)-2))/(24/delta_T)
    
    #% INITIALIZATION
    
    BF=np.zeros(N)
    QS=np.zeros(N)
    WW=np.zeros(N)
    PERC=np.zeros(N)
    
    
    #% MAIN ROUTINE
    P=PIO.values
    #Q=Qobs.values
    
    W=W_p*W_max
    PIOprec=0
    S=np.nan
    Pcum=0
    IE=0
    
    for t in range(N):

        IE=P[t]*(W/W_max)**alpha
        E=EPOT[t]*W/W_max
        PERC=Nu*Ks*(W/W_max)**(m2)
        BF[t]=(1-Nu)*Ks*(W/W_max)**(m2)
        W=W+(P[t]-BF[t]-IE-PERC-E)
        
        # data assimilation with nudging
        
        if K>0:
            if ~np.isnan(Wobs[t]):
                W=K*(Wobs[t]*W_max)+(1-K)*W
        
        
        if W>=W_max:
            SE=W-W_max
            W=W_max
        else:
            SE=0
        
        QS[t]=IE+SE
        WW[t]=W/W_max
        
        if t>2:
            PIOprec=np.sum(P[t-3:t])
    
    
    
    WWW=pd.DataFrame(WW, index=data_input.index)
    WWW.columns=['W']
    df2=data_input.join(WWW)
    
    
    #% Convolution (GIUH)
    IUH1=IUH_comp(gamma1,Ab,dt,delta_T)*dt
    IUH1=IUH1/np.sum(IUH1)
    IUH2=IUH_NASH(1,0.5*gamma1,Ab,dt,delta_T)*dt
    IUH2=IUH2/np.sum(IUH2);
    
    QSint=np.interp(np.arange(0,N,dt),np.arange(0,N,1),QS)
    BFint=np.interp(np.arange(0,N,dt),np.arange(0,N,1),BF)
    
    
    
    temp1=np.convolve(IUH1,QSint)
    temp2=np.convolve(IUH2,BFint)
    
    yy=np.arange(0,N*np.round(1/dt),np.round(1/dt))
    ii=yy.astype(int)
    
    
    
    #Qsim1=temp2[ii]*(Ab*1000./delta_T/3600)
    Qsim=(temp1[ii]+temp2[ii])*(Ab*1000./delta_T/3600)
    
    
    te1 = pd.DataFrame(Qsim, index=data_input.index, columns=list('S'))
    Qout=Qobs.copy()
    te2=pd.Series.to_frame(Qout)
    QQ=te1.join(te2)
    QQ.columns=['S','O']
    df3=df2.join(te1)
    out=Perf(QQ)
    #% PRINT FIGURE
    
    if fig>0:
        stringa_per= name[0:-4]+" NS="+ "%0.3f" % out.NS()+" ANSE="+ "%0.3f" % out.ANSE()+" RMSE="+ "%0.3f" % out.RMSE()+' $ m^3/s$'
        
        f, ax = plt.subplots(2, sharex=True, figsize=(12, 12))
        ax[0].plot(df3.index, df3['P'].values,label='Rainfall',color='b')
        ax[0].set_ylim(0,np.max(df3['P'].values)+5)
        ax[0].set_ylabel('Rainfall [mm]', fontsize=16)
        ax2 = ax[0].twinx()
        ax2.plot(df3.index, df3['W'].values,label='Soil Moisture',color='g')
        ax2.set_title(stringa_per,fontsize=20)
        ax2.set_ylim(0,np.max(df3['W'].values)+0.05)
        ax2.set_ylabel('Relative saturation [-]', fontsize=16) 
        ax[0].grid(True)
        ax[0].tick_params(axis='y', labelsize=16)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.legend(loc='upper right', shadow=True)
        ax[0].legend(loc='upper right', shadow=True)
        
        ax[1].set_ylabel('Rainfall [mm]')
        ax[1].plot(df3.index, df3['Q'].values,label='Qobs',color='g')
        ax[1].plot(df3.index, df3['S'].values,label='Qsim',color='r')
        ax[1].set_ylim(0,np.max(df3.max())+10)
        #ax[1].set_ylim(0,df3.+10)
        ax[1].set_ylabel('Discharge [$m^3/s$]', fontsize=16)
        ax[1].grid(True)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].tick_params(axis='x', labelsize=14)
        ax[1].legend(loc='upper right', shadow=True)
        #plt.legend(loc='lower right') 
        f.savefig(name[0:-4]+'.png',dpi=200)
    
    return out,df3
#%% MODDEL RUN

if __name__ == '__main__':

   name='migi_0406.txt'
   data_input=pd.read_csv(name,index_col=0,header = None, names = ['P','T','Q'])
   PAR=np.loadtxt('X_opt_'+name)
   fig=1
   
   QobsQsim,data=MILC(name,data_input,PAR,fig)
   
   Wmodel=data['W']
   
   ASC=pd.read_csv('ASCAT.csv',index_col=0)

   # here must match the ASCAT time series with those of the model i.e. ASC and Wmodel
   
   QobsQsim,data=MILC(name,data_input,PAR,Ab,fig,ASC,K=0.1)
   
   print(QobsQsim.NS())
   print(QobsQsim.R())
   print(data)
