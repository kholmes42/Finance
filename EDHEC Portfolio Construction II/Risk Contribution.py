# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:27:37 2023
Risk contributions
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize
import importlib

tmr = importlib.import_module("Time Varying Risk Parameters")


#prep data
df_rets = pd.read_csv(r"ind49_m_vw_rets.csv",index_col=0)
df_nfirms = pd.read_csv(r"ind49_m_nfirms.csv",index_col=0)
df_size = pd.read_csv(r"ind49_m_size.csv",index_col=0)

df_rets.index = pd.to_datetime(df_rets.index,format="%Y%m")
df_rets.columns = df_rets.columns.str.rstrip()

df_nfirms.index = pd.to_datetime(df_nfirms.index,format="%Y%m")
df_nfirms.columns = df_nfirms.columns.str.rstrip()

df_size.index = pd.to_datetime(df_size.index,format="%Y%m")
df_size.columns = df_size.columns.str.rstrip()





df_rets = df_rets["1974":].copy()/100
df_nfirms = df_nfirms["1974":].copy()
df_size = df_size["1974":].copy()



#get capweighted weights

df_cw = df_nfirms * df_size
df_cw = df_cw.divide(df_cw.sum(axis=1),axis=0)


wgts_cw = tmr.alloc_cw(df_rets,df_nfirms,df_size)
wgts_ew = tmr.alloc_ew(df_rets,df_nfirms)
wgts_gmv = tmr.alloc_gmv(df_rets,lookback=36,structure="Shrinkage",prior=0.5)
wgts_rp = tmr.alloc_risk_parity(df_rets,lookback=36,structure="Shrinkage",prior=0.5)


    


#graph ENC
fig,ax = plt.subplots(1,3,figsize=(18,5))
enc_cw = tmr.ENC(wgts_cw)
enc_ew = tmr.ENC(wgts_ew)
enc_gmv = tmr.ENC(wgts_gmv)
enc_rp = tmr.ENC(wgts_rp)
sns.lineplot(enc_cw,label="CW",ax=ax[0])
sns.lineplot(enc_ew,label="EW",ax=ax[0])
sns.lineplot(enc_gmv,label="GMV",ax=ax[0])
sns.lineplot(enc_rp,label="Risk Parity",ax=ax[0])
ax[0].set_title("Effective Number of Constituents (ENC)")
ax[0].set_ylabel("ENC")


usecov = tmr.build_rolling_cov_matrix(df_rets,lookback=36,structure=None,prior=0.5)

#calculate ENCB
encb_cw = tmr.rolling_ENCB(wgts_cw,usecov)
encb_ew = tmr.rolling_ENCB(wgts_ew,usecov)
encb_gmv = tmr.rolling_ENCB(wgts_gmv,usecov)
encb_rp = tmr.rolling_ENCB(wgts_rp,usecov)

sns.lineplot(encb_cw,label="CW",ax=ax[1])
sns.lineplot(encb_ew,label="EW",ax=ax[1])
sns.lineplot(encb_gmv,label="GMV",ax=ax[1])
sns.lineplot(encb_rp,label="Risk Parity",ax=ax[1])
ax[1].set_title("Ex-Post Rolling 36 Month Risk")
ax[1].set_ylabel("Standard Deviation")
ax[1].set_title("Effective Number of Correlated Bets (ENCB)")
ax[1].set_ylabel("ENCB")



#calculate rolling expost vol
vol_cw = np.sum(wgts_cw*df_rets,axis=1).rolling(36).std()*np.sqrt(12)
vol_ew = np.sum(wgts_ew*df_rets,axis=1).rolling(36).std()*np.sqrt(12)
vol_gmv = np.sum(wgts_gmv*df_rets,axis=1).rolling(36).std()*np.sqrt(12)
vol_rp = np.sum(wgts_rp*df_rets,axis=1).rolling(36).std()*np.sqrt(12)

sns.lineplot(vol_cw,label="CW",ax=ax[2])
sns.lineplot(vol_ew,label="EW",ax=ax[2])
sns.lineplot(vol_gmv,label="GMV",ax=ax[2])
sns.lineplot(vol_rp,label="Risk Parity",ax=ax[2])

ax[2].set_title("Ex-Post Rolling 36 Month Risk")
ax[2].set_ylabel("Standard Deviation")



#calculate risk contributions
rc_ew, mrc_ew = tmr.r_contributions(wgts_ew.iloc[-1],df_rets.cov())
rc_cw, mrc_cw = tmr.r_contributions(wgts_cw.iloc[-1],df_rets.cov())
rc_ew.sort_values(inplace=True,ascending=False)
rc_cw.sort_values(inplace=True,ascending=False)

fig,ax = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Top 10 Risk Contribution by Industry")
sns.barplot(x=rc_cw[:10].index,y=rc_cw[:10],ax=ax[0],color="blue")
sns.barplot(x=rc_ew[:10].index,y=rc_ew[:10],ax=ax[1],color="blue")
ax[0].set_title("Cap Weighted")
ax[0].set_ylabel("Risk Contribution %")
ax[1].set_title("Equal Weighted")
ax[1].set_ylabel("Risk Contribution %")


