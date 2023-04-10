# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:56:07 2023
EDHEC Alt Data Week 1
@author: kholm
dataset: https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city?resource=download
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from math import radians, cos, sin, asin, sqrt
from scipy import stats


df = pd.read_csv(r"uber-raw-data-jul14.csv")

print(df.info())
print(df.head())

#shorten for testing
#df = df.iloc[:10000,:]


df["Date/Time"] = pd.to_datetime(df["Date/Time"]).dt.floor("1H")

###
#Look at heatmap of times and days
###

df["Day"] = df["Date/Time"].apply(lambda x: calendar.day_name[calendar.weekday(x.year,x.month,x.day)])
df["Hour"] = df["Date/Time"].apply(lambda x: x.hour)



dfgrouped = df.groupby(["Day","Hour"]).size().reset_index(name='Counts')


dfpiv = dfgrouped.pivot(columns=["Day"],index=["Hour"],values=["Counts"]).droplevel(None,axis=1)
dfpiv = dfpiv.reindex(columns=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])


fig,ax = plt.subplots(1,2,figsize=(18,7))

sns.heatmap(dfpiv,cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),ax=ax[0])
ax[0].set_title("NYC Uber Trip Most Frequent Heatmap")



###
#check big attractions in NYC
###

metro_coord = (40.7794,-73.9632)
empirestate_coord = (40.7484,-73.9857)
yankstadium_coord = (40.8296,-73.9262)
oneworld_coord = (40.7127,-74.0134)


def haversine(coor1, coor2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    SOURCE: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [coor1[0], coor1[1], coor2[0], coor2[1]])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r



df["wkday"] = df["Date/Time"].apply(lambda x: calendar.weekday(x.year,x.month,x.day))
df["date"] =  pd.to_datetime(df['Date/Time']).dt.date
df["Loc"] = df[["Lat","Lon"]].apply(lambda x: tuple(x),axis=1)

df["DisttoMet"] = df["Loc"].apply(haversine,coor2=metro_coord)
df["DisttoEmpireBld"] = df["Loc"].apply(haversine,coor2=empirestate_coord)
df["DisttoYankees"] = df["Loc"].apply(haversine,coor2=yankstadium_coord)
df["DisttoWorldTrd"] = df["Loc"].apply(haversine,coor2=oneworld_coord)


df2 = df.filter(regex='Distto')

print(df2[df2 < 0.25].count())


### 
#check differences in weekend vs weekday
###

df3 = df[["Date/Time","Hour","date"]].groupby(["Date/Time","date"]).count().reset_index()

df3["wgt"] = df3.groupby('date')['Hour'].transform(lambda x: x/x.sum())



tstats = []
for i in range(0,24):
   wkday = df3[(df3["Date/Time"].dt.hour == i) & (df3["Date/Time"].dt.weekday <= 4)]["wgt"]
   wkend = df3[(df3["Date/Time"].dt.hour == i) & (df3["Date/Time"].dt.weekday > 4)]["wgt"]
   tstats.append(stats.ttest_ind(wkday,wkend)[0])



gp = sns.barplot(x=np.arange(0,24),y=tstats,color="b",ax=ax[1])
gp.axhline(1.96, linestyle='--')
gp.axhline(-1.96, linestyle='--')
ax[1].set_title("Hourly Difference T-Stat Test Weekday vs Weekend")
ax[1].set_ylabel("T-Stat")
ax[1].set_xlabel("Hour")