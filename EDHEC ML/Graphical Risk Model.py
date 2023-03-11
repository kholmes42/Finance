# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 20:05:02 2023
HW Week 3
@author: kholm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import MDS


np.random.seed(42)

df_rets = pd.read_csv(r"C:/Users/kholm/OneDrive/Documents/RESUMES/Full Time/GT MS Analytics/Coursera/EDHEC Portfolio Construction II/ind49_m_vw_rets.csv",index_col=0)


df_rets.index = pd.to_datetime(df_rets.index,format="%Y%m")
df_rets.columns = df_rets.columns.str.rstrip()


df_r = df_rets["2010":].copy()/100
df_r /= df_r.std(axis=0)


#create sparse connections
mod = GraphicalLassoCV(max_iter=1000).fit(df_r)
precision_mat = mod.get_precision()

#cluster covariance matrix in high dim space
clustering = AffinityPropagation().fit(mod.covariance_)

#create 2D embedding for visualization
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(df_r.T)

plt.figure(figsize=(12,12))
ax = sns.scatterplot(x=X_transformed[:,0],y=X_transformed[:,1],hue=clustering.labels_,
                palette = sns.color_palette("tab10"),s=1000)


#add connections from precision matrix (conditional independence)
for i in range(0,precision_mat.shape[0]):
    for j in range(0,precision_mat.shape[1]):
        if i != j:
            if np.abs(precision_mat[i][j]) > .03:
                
                xloc1 = X_transformed[i][0]
                xloc2 = X_transformed[j][0]
                yloc1 = X_transformed[i][1]
                yloc2 = X_transformed[j][1]
                
                plt.plot([xloc1, xloc2], [yloc1, yloc2], linewidth=0.15)

plt.xticks([])
plt.yticks([])
plt.title("Graphical Network of Fama-French 49 Industries 2010-2018")

#describe clusters
for k in np.unique(clustering.labels_):
    print(k)
    print(df_r.columns[clustering.labels_==k])
    
legend_handles, _= ax.get_legend_handles_labels()
ax.legend(legend_handles, ["Agriculture","Consumable Staples","Discretionary Non-Durable","Defensives","Industrials","Guns","Gold","Heavy Industrials","Electronics","Services"], bbox_to_anchor=(1,1))