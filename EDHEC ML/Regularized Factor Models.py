# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:58:12 2023
Exploring Regularized Factor Models
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
import datetime

df = pd.read_csv("Data_Oct2018_v2.csv",index_col = 0)
df2 = pd.read_csv("Assets_7.csv",index_col = 0)

print(df.head())
print(df.info())

df.index = pd.to_datetime(df.index).date
df2.index = pd.to_datetime(df2.index).date

xvars = ["World Equities","US Treasuries","Bond Risk Premium","Inflation Protection","Currency Protection"]
yvars = ["US Equities","Real Estate","Commodities","Corp Bonds"]

dfy = df[yvars].copy()
dfx = df[xvars].copy()


dfx = dfx[datetime.date(1997, 1, 1):datetime.date(2014, 12, 31)]
dfy = dfy[datetime.date(1997, 1, 1):datetime.date(2014, 12, 31)]


df_linear_coef = pd.DataFrame(index=yvars,columns = xvars)
df_ridge_coef = pd.DataFrame(index=yvars,columns = xvars)
df_lasso_coef = pd.DataFrame(index=yvars,columns = xvars)

for asset in yvars:
    
    #run simplified Linear factor model
    lin_factor = LinearRegression()
    lin_factor.fit(dfx,dfy[asset])
    df_linear_coef.loc[asset] = lin_factor.coef_
    
    
    #run L2 regularized linear factor model
    ridge_factor = RidgeCV()
    ridge_factor.fit(dfx,dfy[asset])
    df_ridge_coef.loc[asset] = ridge_factor.coef_
    
    
    #run L1 regularized linear factor model
    lasso_factor = Lasso(alpha=0.0002)
    lasso_factor.fit(dfx,dfy[asset])
    df_lasso_coef.loc[asset] = lasso_factor.coef_


colors = sns.diverging_palette(10, 133, as_cmap=True)

fig,ax = plt.subplots(1,2,figsize=(18,6))
sns.heatmap(df_linear_coef.astype(float),ax=ax[0],annot=True,cmap=colors,cbar=False)
ax[0].set_title("Linear Factor Model Betas")
# sns.heatmap(df_ridge_coef.astype(float),ax=ax[1],annot=True,cmap=colors,cbar=False)
# ax[1].set_title("Ridge Factor Model Betas")
sns.heatmap(df_lasso_coef.astype(float),ax=ax[1],annot=True,cmap=colors,mask=(df_lasso_coef==0),cbar=False)
ax[1].set_title("Lasso Factor Model Betas")