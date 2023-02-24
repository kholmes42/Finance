# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:33:10 2023
Week 3 Analyzing Limits of Diversification
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
np.set_printoptions(suppress=True)


dfsize = pd.read_csv(r"ind30_m_size.csv",index_col=0)
dfnfirms = pd.read_csv(r"ind30_m_nfirms.csv",index_col=0)
dfrets = pd.read_csv(r"ind30_m_vw_rets.csv",index_col=0)/100
dfrets.columns = dfrets.columns.str.rstrip()



dfsize.index = pd.to_datetime(dfsize.index,format="%Y%m")
dfnfirms.index = pd.to_datetime(dfnfirms.index,format="%Y%m")
dfrets.index = pd.to_datetime(dfrets.index,format="%Y%m")



#calculate mkt cap weights by industry
dfmcap = dfsize*dfnfirms
df_mcap_wgt = dfmcap.div(np.sum(dfmcap,axis=1),axis=0)
df_mcap_wgt.columns = df_mcap_wgt.columns.str.rstrip()

#examine changing industry weights
plt.figure()
df_mcap_wgt[["Fin","Steel"]].plot()
plt.ylabel("Weight in Market")
#

df_mcap_ret = np.sum(df_mcap_wgt*dfrets,axis=1)

df_mc_index = np.cumprod(df_mcap_ret+1)*100

plt.figure()
sns.lineplot(df_mc_index["1980":])
sns.lineplot(df_mc_index["1980":].rolling(window=36).mean(),label="Trailing 36m avg")

plt.title("Market Return 1926-2018")
plt.ylabel("Index Level")


plt.figure(figsize=(12,6))
sns.lineplot(df_mcap_ret.rolling(window=36).mean()*12,label="Trailing 36 Month Ret Annualized")
sns.lineplot(df_mcap_ret,label="Monthly Return")
plt.ylabel("Return")


dfrollcorr = dfrets.rolling(window=36).corr()
dfrollcorr.index.names = ["date","industry"]

avgrollingcorr = dfrollcorr.groupby(level="date").apply(lambda corm: corm.values[np.triu_indices_from(corm.values,1)].mean())
ax2 = plt.twinx()
sns.lineplot(avgrollingcorr,ax=ax2,color="green")
ax2.set_ylabel("Correlation (green)")




#CPPI
startlvl = 100
floor = 0.8
m = 3

risky_r = pd.DataFrame(df_mcap_ret["2000":])
risky_r.columns = ["ret"]
rf_r = pd.DataFrame().reindex_like(risky_r)
rf_r[:] = 0.03/12
acc_history = pd.DataFrame(index=rf_r.index,columns = ["Total","Risky","Riskless","Floor"])

acc_history.iloc[0]["Total"] = startlvl
acc_history.iloc[0]["Risky"] = m*(startlvl-floor*startlvl)
acc_history.iloc[0]["Riskless"] = startlvl - acc_history.iloc[0]["Risky"]
acc_history.iloc[0]["Floor"] = startlvl*(floor)
for i in range(len(risky_r)-1):
    riskyval = (acc_history.iloc[i]["Risky"])*(1+risky_r.iloc[i]["ret"])
    risklessval = (acc_history.iloc[i]["Riskless"])*(1+rf_r.iloc[i]["ret"])
    newportval = riskyval + risklessval
    
    newriskalloc = np.minimum(np.maximum(m*(newportval - startlvl*floor),0),newportval)

    acc_history.iloc[i+1]["Risky"] =  newriskalloc
    acc_history.iloc[i+1]["Riskless"] = newportval - newriskalloc
    acc_history.iloc[i+1]["Total"] = acc_history.iloc[i+1]["Risky"] + acc_history.iloc[i+1]["Riskless"] 
    acc_history.iloc[i + 1]["Floor"] = startlvl*(floor)


fig, ax = plt.subplots(2, 2,figsize=(18,10))
acc_history.plot(ax=ax[0][0])
ax[0][0].set_title("CPPI Breakdown")

sns.lineplot(np.cumprod(risky_r+1)*startlvl,ax=ax[0][1],label="Market")
sns.lineplot(acc_history["Total"] ,ax=ax[0][1],label="CPPI Strategy")
ax[0][1].set_title("CPPI Strategy")


a = Line2D([], [], color='blue', label='Market')
b = Line2D([], [], color='orange', label='CPPI')

ax[0][1].legend(handles=[a, b])
ax[0][1].set_xlim(risky_r.index.min())





#CPPI w Max DD
startlvl = 100
floor = 0.8
floorlvl = floor * startlvl
m = 3
peak = startlvl

risky_r = pd.DataFrame(df_mcap_ret["2000":])
risky_r.columns = ["ret"]
rf_r = pd.DataFrame().reindex_like(risky_r)
rf_r[:] = 0.03/12
acc_history = pd.DataFrame(index=rf_r.index,columns = ["Total","Risky","Riskless","Floor"])

acc_history.iloc[0]["Total"] = startlvl
acc_history.iloc[0]["Risky"] = m*(startlvl-floor*startlvl)
acc_history.iloc[0]["Riskless"] = startlvl - acc_history.iloc[0]["Risky"]
acc_history.iloc[0]["Floor"] = startlvl*(floor)
for i in range(len(risky_r)-1):

    riskyval = (acc_history.iloc[i]["Risky"])*(1+risky_r.iloc[i]["ret"])
    risklessval = (acc_history.iloc[i]["Riskless"])*(1+rf_r.iloc[i]["ret"])
    newportval = riskyval + risklessval
    
    if newportval > peak:
        floorlvl = peak*(floor)
        peak = newportval

    else:
        floorlvl = floorlvl
    
    newriskalloc = np.minimum(np.maximum(m*(newportval - floorlvl),0),newportval)

    acc_history.iloc[i+1]["Risky"] =  newriskalloc
    acc_history.iloc[i+1]["Riskless"] = newportval - newriskalloc
    acc_history.iloc[i+1]["Total"] = acc_history.iloc[i+1]["Risky"] + acc_history.iloc[i+1]["Riskless"] 
    acc_history.iloc[i + 1]["Floor"] = floorlvl



acc_history.plot(ax=ax[1][0])
ax[1][0].set_title("CPPI w Max DD Breakdown")

sns.lineplot(np.cumprod(risky_r+1)*startlvl,ax=ax[1][1],label="Market")
sns.lineplot(acc_history["Total"] ,ax=ax[1][1],label="CPPI Strategy")
ax[1][1].set_title("CPPI w Max DD Strategy")


a = Line2D([], [], color='blue', label='Market')
b = Line2D([], [], color='orange', label='CPPI')

ax[1][1].legend(handles=[a, b])
ax[1][1].set_xlim(risky_r.index.min())





def gbm(periods = 10,steps_per_period=12, n=1000,mu = 0.07,sigma=0.15):
    """
    Use simple GBM to generate price returns
    """
  
    r_drift = np.random.normal(1+mu/steps_per_period, sigma*np.sqrt(1/steps_per_period),size=(periods*steps_per_period,n))
    prices = 100*pd.DataFrame(( r_drift )).cumprod()
    
    return prices


prices = gbm(n=100)
plt.figure()
prices.plot(legend=False)