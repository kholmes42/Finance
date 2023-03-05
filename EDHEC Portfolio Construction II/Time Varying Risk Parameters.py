# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:57:21 2023
Time Varying Risk parameters
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import arch
import cvxpy as cp
import SummaryPerformance as perf
from scipy.optimize import minimize

np.set_printoptions(suppress=True)

def r_contributions(w,cov):
    """
    calculates risk contributions
    """

    portvar = w.T@cov@w
  

    marginal_contribution = cov@w
    risk_contribution = np.multiply(marginal_contribution,w.T)/portvar
    
    return risk_contribution,marginal_contribution



def ENC(wgts):
    """
    Calculates effective number of constituents (ENC)
    """

    if wgts.ndim == 1:
        enb = np.sum(wgts**2)**-1
    else:
        enb = np.sum(wgts**2,axis=1)**-1
    
    return enb



def rolling_ENCB(w,cov):
    """
    calcualte rolling ENCB
    """
    cov2 = cov.dropna()
    df_encb = pd.DataFrame(index=cov2.index.get_level_values(0).unique(),columns=["encb"])
    w2 = w[cov2.index.get_level_values(0).min():].copy()
  
    for i in w2.index:
        df_encb.loc[i] = ENC(r_contributions(w2.loc[i],cov2.loc[i])[0])
    
    
    
    return df_encb.squeeze()



def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = r_contributions(weights, cov)[0]
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x



"""

Structuring the Covariance Matrix

"""


def bt_mix(rets,allocator,**kargs):
    """
    create time series that combines returns
    """
    
    wgts = allocator(rets,**kargs)
    r_mix = wgts*rets
    
    return np.sum(r_mix,axis=1),wgts






def alloc_cw(rets,df_firms,df_size):
    """
    calculate cap weighting
    """
    
    df_cw = df_firms * df_size
    df_cw = df_cw.divide(df_cw.sum(axis=1),axis=0)
    
    return df_cw



def alloc_ew(rets,df_firms):
    """
    calculate equal weighting
    """
    
    df_ew = pd.DataFrame().reindex_like(df_firms)
    df_ew.iloc[:,:] = 1/len(df_ew.columns)

    return df_ew



def build_rolling_cov_matrix(rets,lookback=36,structure=None,prior=0.5):
   
    #no structure, use simple rolling lookback covariance matrix
    if structure == None:
        dfrolledcov = rets.rolling(window=lookback).cov().dropna()
     
    #shrink to constant correlation
    elif structure == "Constant Correlation" or structure == "Shrinkage":
        dfrolledcorr = rets.rolling(window=lookback).corr().dropna()
        dfrolledstd = rets.rolling(window=lookback).std().dropna()
        n = dfrolledcorr.iloc[0].shape[0]
      
        dfrolledcov = pd.DataFrame().reindex_like(dfrolledcorr)

        for i in dfrolledcorr.index.get_level_values(0).unique():
            correl = dfrolledcorr.loc[i]
            rbar = (dfrolledcorr.loc[i].values.sum()-n)/(n*(n-1))
            ccor = np.full_like(correl,rbar)
            np.fill_diagonal(ccor,1.)
            sd = dfrolledstd.loc[i]
        
            dfrolledcov.loc[i] = ccor*np.outer(sd,sd)
        
        #add back sample
        if structure == "Shrinkage":
            dfrolledcov = dfrolledcov*prior + (1-prior)*rets.rolling(window=lookback).cov().dropna()
        
        
    else:
        print("Not implemented yet")
    
    return dfrolledcov


def alloc_gmv(rets,lookback=36,structure=None,prior=0.5):
    """
    calculate GMV weighting
    """
    assert structure in [None,"Constant Correlation","Shrinkage"], "Structure not avaialble."
    
    df_gmv = pd.DataFrame().reindex_like(rets)
    
    dfrolledcov = build_rolling_cov_matrix(rets,lookback=lookback,structure=structure,prior=prior)
        
        
    #SOLVE OPTIMIZATION PROBLEM FOR EACH REBAL
    for i in dfrolledcov.index.get_level_values(0).unique():
   
        
        covm = dfrolledcov.loc[i]
     
        w = cp.Variable(len(rets.columns))
    
        #calcualte GMV
        objective = cp.Minimize(0.5*cp.quad_form(w,covm))
        constraints = [w >= 0, #long only
                        cp.sum(w) == 1,
                        w <= 0.33] #100% invested
                 
                      
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
    
        df_gmv.loc[i] = w.value
    

    return df_gmv



def alloc_risk_parity(rets,tgt_risk=None,lookback=36,structure=None,prior=0.5):
    """
    calculate Risk Parity weighting
    """
    assert structure in [None,"Constant Correlation","Shrinkage"], "Structure not avaialble."
    
    #if no target risk given assume equal risk contribtution
    if tgt_risk is None:
        tgt_risk = np.repeat(1/len(rets.columns),len(rets.columns))
    
    df_rp = pd.DataFrame().reindex_like(rets)
    
    dfrolledcov = build_rolling_cov_matrix(rets,lookback=lookback,structure=structure,prior=prior)
    
         
    #SOLVE OPTIMIZATION PROBLEM FOR EACH REBAL
    for i in dfrolledcov.index.get_level_values(0).unique():
   
        
        covm = dfrolledcov.loc[i]
     
        w = target_risk_contributions(tgt_risk,covm)
        
        df_rp.loc[i] = w
    
    
    return df_rp




if __name__ == "main":

    df_factors = pd.read_csv(r"F-F_Research_Data_Factors_m.csv",index_col=0)
    
    
    df_factors.index = pd.to_datetime(df_factors.index,format="%Y%m")
    
    df_factors.columns =df_factors.columns.str.rstrip()
    
    
    df_mkt = pd.DataFrame(df_factors["Mkt-RF"].copy()/100)
    
    dfvol = df_mkt.copy()
    sp = 20
    dfvol["Mkt-RF"] = df_mkt["Mkt-RF"].rolling(12).std()*np.sqrt(12)
    dfvol["EWMA (Alp=" + str(round(2/(sp+1),2)) + ")"] = df_mkt["Mkt-RF"].ewm(span=sp).std()*np.sqrt(12)
    
    
    garch = arch.arch_model(df_mkt["Mkt-RF"]*10, vol='garch', p=1, o=0, q=1)
    garch_fitted = garch.fit()
    
    dfvol["GARCH(1,1)"] = garch_fitted._volatility/10*np.sqrt(12)
    
    
    dfvol.dropna(inplace=True)
    
    
    
    dfvol["mean"] = dfvol["Mkt-RF"].mean()
    dfvol.rename(columns={"Mkt-RF":"Simple"},inplace=True)
    plt.figure()
    sns.lineplot(dfvol)
    plt.xlim(dfvol.index.min(),dfvol.index.max())
    plt.title("Rolling Volatility of Market")
    plt.ylabel("Volatility")
    
    
    
    
    
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
    
    #get equal weights
    df_ew = pd.DataFrame().reindex_like(df_cw)
    df_ew.iloc[:,:] = 1/len(df_ew.columns)
    
    
    
    
    #run backtests
    gmvCCrets,ccwgt =  bt_mix(df_rets, alloc_gmv,structure="Constant Correlation")
    gmvsamplerets,samwgt =  bt_mix(df_rets, alloc_gmv)
    gmvShrinkrets,swgt =  bt_mix(df_rets, alloc_gmv,structure="Shrinkage")
    ewrets,ewwgt = bt_mix(df_rets[gmvsamplerets.index.min():],alloc_ew,df_firms=df_nfirms[gmvsamplerets.index.min():])
    cwrets,cwwgt = bt_mix(df_rets[gmvsamplerets.index.min():],alloc_cw,df_firms=df_nfirms[gmvsamplerets.index.min():],df_size=df_size[gmvsamplerets.index.min():])
    gmvsamplerets.name = "GMV-Sample"
    gmvCCrets.name = "GMV-Const Correl"
    gmvShrinkrets.name = "GMV-Shrinkage"
    ewrets.name = "EW-Industries"
    cwrets.name = "CW"
    
    
    #graph cumulative returns
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    sns.lineplot((1+ewrets).cumprod(),label=ewrets.name,ax=ax[0])
    sns.lineplot((1+cwrets).cumprod(),label=cwrets.name,ax=ax[0])
    sns.lineplot((1+gmvsamplerets).cumprod(),label=gmvsamplerets.name,ax=ax[0])
    sns.lineplot((1+gmvCCrets).cumprod(),label=gmvCCrets.name,ax=ax[0])
    sns.lineplot((1+gmvShrinkrets).cumprod(),label=gmvShrinkrets.name,ax=ax[0])
    ax[0].set_xlim(ewrets.index.min(),ewrets.index.max())
    ax[0].set_ylabel("Index Level")
    ax[0].set_title("Fama French Industry Portfolios")
    
    print()
    
    timeseries = pd.concat([cwrets,ewrets,gmvsamplerets,gmvCCrets,gmvShrinkrets],axis=1)
    
    print(perf.summary_stats(timeseries))
    
    
    #show weights
    print()
    dfallwgts = pd.concat([cwwgt.iloc[-1],ewwgt.iloc[-1],samwgt.iloc[-1],swgt.iloc[-1],ccwgt.iloc[-1]],axis=1).T
    dfallwgts.index = ["CW","EW","GMV-Sample","GMV-Shrinkage","GMV-Constant Corr"]
    
    dfallwgts.plot(kind = 'bar',stacked=True,legend=False,ax=ax[1])
    
    ax[1].set_ylabel("Allocation %")
    ax[1].set_xlabel("Portfolio")
    ax[1].set_title("Portfolio Weighting Schemes Industry Allocation")
