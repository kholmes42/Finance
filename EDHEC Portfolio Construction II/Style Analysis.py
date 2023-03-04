# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 20:47:36 2023
Rolling Style Analysis
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.linear_model import LinearRegression

np.set_printoptions(suppress=True)

df_factors = pd.read_csv(r"F-F_Research_Data_Factors_m.csv",index_col=0)
df_rets = pd.read_csv(r"ind49_m_vw_rets.csv",index_col=0)

df_factors.index = pd.to_datetime(df_factors.index,format="%Y%m")
df_rets.index = pd.to_datetime(df_rets.index,format="%Y%m")
df_factors.columns =df_factors.columns.str.rstrip()
df_rets.columns = df_rets.columns.str.rstrip()


df_f = df_factors["1991":].copy()
df_r = df_rets["1991":].copy()



#q1 Get CAPM Beta Beer

mod = LinearRegression()

mod.fit(np.array(df_f["Mkt-RF"]).reshape(-1, 1),np.array(df_r["Beer"] - df_f["RF"]).reshape(-1, 1))
print(mod.coef_)


#q2 Get CAPM Beta Steel

mod = LinearRegression()

mod.fit(np.array(df_f["Mkt-RF"]).reshape(-1, 1),np.array(df_r["Steel"]- df_f["RF"]).reshape(-1, 1))
print(mod.coef_)


df_f = df_factors["2013":].copy()
df_r = df_rets["2013":].copy()



#q3 Get CAPM Beta Beer

mod = LinearRegression()

mod.fit(np.array(df_f["Mkt-RF"]).reshape(-1, 1),np.array(df_r["Beer"]- df_f["RF"]).reshape(-1, 1))
print(mod.coef_)


#q4 Get CAPM Beta Steel

mod = LinearRegression()

mod.fit(np.array(df_f["Mkt-RF"]).reshape(-1, 1),np.array(df_r["Steel"]- df_f["RF"]).reshape(-1, 1))
print(mod.coef_)


#q5 get highest beta

df_f = df_factors["1991":"1993"].copy()
df_r = df_rets["1991":"1993"].copy()


mbeta = -99
mind = ""
for ind in df_r.columns:
    mod = LinearRegression()
    mod.fit(np.array(df_f["Mkt-RF"]).reshape(-1, 1),np.array(df_r[ind]- df_f["RF"]).reshape(-1, 1))
    if mod.coef_[0] > mbeta:
        mbeta = mod.coef_[0]
        mind = ind
    
print(mind)
print(mbeta)




#q6 get highest beta

df_f = df_factors["1991":"1993"].copy()
df_r = df_rets["1991":"1993"].copy()


mbeta = 99
mind = ""
for ind in df_r.columns:
    mod = LinearRegression()
    mod.fit(np.array(df_f["Mkt-RF"]).reshape(-1, 1),np.array(df_r[ind]- df_f["RF"]).reshape(-1, 1))
    if mod.coef_[0] < mbeta:
        mbeta = mod.coef_[0]
        mind = ind
    
print(mind)
print(mbeta)



#q7 largest Sc tilt
df_f = df_factors["1991":].copy()
df_r = df_rets["1991":].copy()


mbeta = -99
mind = ""
for ind in df_r.columns:
    mod = LinearRegression()
    mod.fit(np.array(df_f[["Mkt-RF","SMB","HML"]]),np.array(df_r[ind]- df_f["RF"]).reshape(-1, 1))
    if mod.coef_[0][1] > mbeta:
        mbeta = mod.coef_[0][1]
        mind = ind
    
print(mind)
print(mbeta)





#q8 smallest Sc tilt
df_f = df_factors["1991":].copy()
df_r = df_rets["1991":].copy()


mbeta = 99
mind = ""
for ind in df_r.columns:
    mod = LinearRegression()
    mod.fit(np.array(df_f[["Mkt-RF","SMB","HML"]]),np.array(df_r[ind]- df_f["RF"]).reshape(-1, 1))
    if mod.coef_[0][1] < mbeta:
        mbeta = mod.coef_[0][1]
        mind = ind
    
print(mind)
print(mbeta)





#q9 largest value tilt
df_f = df_factors["1991":].copy()
df_r = df_rets["1991":].copy()


mbeta = -99
mind = ""
for ind in df_r.columns:
    mod = LinearRegression()
    mod.fit(np.array(df_f[["Mkt-RF","SMB","HML"]]),np.array(df_r[ind]- df_f["RF"]).reshape(-1, 1))
    if mod.coef_[0][2] > mbeta:
        mbeta = mod.coef_[0][2]
        mind = ind
    
print(mind)
print(mbeta)



#q10 smallest value tilt
df_f = df_factors["1991":].copy()
df_r = df_rets["1991":].copy()


mbeta = 99
mind = ""
for ind in df_r.columns:
    mod = LinearRegression()
    mod.fit(np.array(df_f[["Mkt-RF","SMB","HML"]]),np.array(df_r[ind]- df_f["RF"]).reshape(-1, 1))
    if mod.coef_[0][2] < mbeta:
        mbeta = mod.coef_[0][2]
        mind = ind
    
print(mind)
print(mbeta)





def style_analysis(tgt_rets, fact_rets,bmwgts):
    """
    performs simple constrained style analysis
    """ 
    w = cp.Variable(len(bmwgts))
    
    df_data = pd.concat([tgt_rets,fact_rets],axis=1)
    
    covm = df_data.cov()
    

    objective = cp.Minimize(cp.quad_form(w-bmwgts,covm))
    constraints = [w + bmwgts >= 0, #long only
                   cp.sum(w) == 1, #100% invested
                  w[0] ==0 ] #no investing in the target portfolio
                  
    
    
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    

    
    return w.value

industry = "Oil"
lookback = 36
bm_weights = [1,0,0,0]
print(style_analysis(df_r[industry]/100- df_f["RF"]/100, df_f[["Mkt-RF","SMB","HML"]]/100,bm_weights))


df_style = pd.DataFrame(index=df_f.index,columns=df_f.columns[:-1])

#perform rolling style analysis
for i in range(lookback,len(df_r)):
    df_style.iloc[i] = np.around(style_analysis(df_r[industry].iloc[i-36:i]/100- df_f["RF"].iloc[i-36:i]/100, df_f[["Mkt-RF","SMB","HML"]].iloc[i-36:i]/100,bm_weights)[1:],2)
    
    
    
    
plt.figure()   
df_style.dropna().plot.area()
plt.title("Rolling " +str(lookback) + " Month Style Analysis for " + industry + " Industry")
plt.ylim(0,1)
plt.ylabel("Allocation %")
plt.xlabel("Date")