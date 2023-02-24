# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 20:13:44 2023
Week 2 basic Portfolio Optimization
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cvxpy as cp


df = pd.read_csv(r"ind30_m_vw_rets.csv",index_col=0)/100

df.index = pd.to_datetime(df.index,format="%Y%m")
df.columns = df.columns.str.rstrip()
print(df.head())



def ann_ret(data,pers=12):
    
    return ((np.array(np.cumprod(1+(data)))[-1])**(pers/len(data))-1)



dfret = df["1996":"2000"].apply(ann_ret)
cov = df["1996":"2000"].cov()



minret = np.min(dfret)
maxret = np.max(dfret)


rets = np.linspace(minret,maxret,40)

risk = []

rf = 0.15
msr = -99

#create efficient frontier
for r in rets:
#create weights for each asset
    w = cp.Variable(len(df.columns))
    constraints = []
    
    objective = cp.Minimize((1/2)*cp.quad_form(w, cov))
                     # [G @ x <= h,
                     #  A @ x == b])
                     
                    
    constraints.append(w >= 0) #non neg
    constraints.append(cp.sum(w) == 1) #total invested
    constraints.append(w.T@dfret == r) #min ret
    
    prob = cp.Problem(objective,constraints)
    prob.solve()
    
    risk.append(np.sqrt(w.value.T@cov@w.value))
    
    #get MSR port
    if (w.value.T@dfret - rf) / np.sqrt(w.value.T@cov@w.value) > msr:
        msr = (w.value.T@dfret - rf) / np.sqrt(w.value.T@cov@w.value)
        maxsr = w.value.T@dfret
        maxsv = np.sqrt(w.value.T@cov@w.value)
        
    
    
plt.figure()
sns.scatterplot(x=risk,y=rets,label="Frontier")
plt.xlabel("Risk $\sigma$")
plt.ylabel("Return")
plt.title("Risk vs Return FF 30 Industries")
plt.xlim(0,0.15)

plt.plot([0,maxsv],[rf,maxsr],linestyle="dashed",color="red")
sns.scatterplot(x=[maxsv],y=[maxsr],color="red",label="CML", s=100)



#plot GMV
w = cp.Variable(len(df.columns))
constraints = []

objective = cp.Minimize((1/2)*cp.quad_form(w, cov))
                 # [G @ x <= h,
                 #  A @ x == b])
                       
constraints.append(w >= 0) #non neg
constraints.append(cp.sum(w) == 1) #total invested

prob = cp.Problem(objective,constraints)
prob.solve()

sns.scatterplot(x=[np.sqrt(w.value.T@cov@w.value)],y=[w.value.T@dfret],color="orange",label="GMV", s=100)


#plot EW
w = np.ones(len(df.columns))/len(df.columns)


sns.scatterplot(x=[np.sqrt(w.T@cov@w)],y=[w.T@dfret],color="green",label="EW", s=100)



