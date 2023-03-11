# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:47:31 2023
Simple Factor Risk Model
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta


bktst = __import__("Time Varying Risk Parameters")



df_rets = pd.read_csv(r"C:/Users/kholm/OneDrive/Documents/RESUMES/Full Time/GT MS Analytics/Coursera/EDHEC Portfolio Construction II/ind49_m_vw_rets.csv",index_col=0)


df_rets.index = pd.to_datetime(df_rets.index,format="%Y%m")
df_rets.columns = df_rets.columns.str.rstrip()



df_factors = pd.read_csv(r"C:/Users/kholm/OneDrive/Documents/RESUMES/Full Time/GT MS Analytics/Coursera/EDHEC Portfolio Construction II/F-F_Research_Data_Factors_m.csv",index_col=0)


df_factors.index = pd.to_datetime(df_factors.index,format="%Y%m")
df_factors.columns =df_factors.columns.str.rstrip()

df_rets = df_rets/100
df_factors = df_factors/100


st = "1974"
lookback = 36

dfr = df_rets[st:].copy()
dff = df_factors[st:].copy()

dfwgts = dfr.copy()
dfwgts[:] = 1/len(dfwgts.columns)

dfriskbreakdown = pd.DataFrame(index=dfwgts.index,columns = ["Historic Risk","Factor Risk","Specific Risk"])


# for each rebalance period
for i in range(lookback,len(dfr)):
    print(i)
    betas = np.zeros((len(df_rets.columns),3))
    
    cnt = 0
    specific_risk = np.zeros(len(df_rets.columns))
    #for each asset run regression
    for asset in dfr.columns:
     
        mod = LinearRegression()
        
        #get lookback data
        r_asset = dfr[asset].iloc[i-lookback:i] - dff["RF"].iloc[i-lookback:i] #excess asset return
        factor_return  = dff[["Mkt-RF","SMB","HML"]].iloc[i-lookback:i]
        
        
        #fit model
        mod.fit(factor_return,r_asset)
        
        betas[cnt] = mod.coef_
        specific_risk[cnt] = np.var(mod.predict(factor_return) - r_asset)
        
        cnt += 1
        
    #get covariance of factors
    covF = factor_return.cov()
    #build specific risk diagonal matrix
    specMat = np.diag(specific_risk)
    

    total_cov_mat = betas@covF@betas.T + specMat
    dfriskbreakdown.iloc[i]["Factor Risk"] = np.sqrt(dfwgts.iloc[i].T@betas@covF@betas.T@dfwgts.iloc[i])*np.sqrt(12)  
    dfriskbreakdown.iloc[i]["Specific Risk"] = np.sqrt(dfwgts.iloc[i].T@specMat@dfwgts.iloc[i])*np.sqrt(12)  


    
dfriskbreakdown["Historic Risk"] =  np.sum(dfwgts * dfr,axis=1).rolling(36).std()*np.sqrt(12)  
 
dfriskbreakdown.dropna(inplace=True)
   
plt.figure()
sns.lineplot(dfriskbreakdown[["Historic Risk","Factor Risk"]])
plt.title("EW 49 Industry Risk Breakdown")
plt.ylabel("Risk")
plt.xlabel("Date")