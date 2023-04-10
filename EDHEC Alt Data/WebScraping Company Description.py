# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:57:35 2023
EDHEC Alt Data Week 2
@author: kholm
dataset: Wikipedia https://en.wikipedia.org/wiki/S%26P_100
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import requests
import bs4 as bs

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

readnew = True

if readnew == True:

    stemmer = nltk.stem.SnowballStemmer("english")    


    page = requests.get('https://en.wikipedia.org/wiki/S%26P_100').text
    soup = bs.BeautifulSoup(page, 'html.parser')
    table = soup.find('table', class_="wikitable sortable")
    
    ticker = []
    links = []
    sectors = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        tick = tds[0].text.strip()
        sector = tds[2].text.strip()
        if tds[1].find("a") is not None:
            lnk = "https://en.wikipedia.org/" + tds[1].find("a").get("href").strip()
            ticker.append(tick)
            links.append(lnk)
            sectors.append(sector)
            
            
    df = pd.DataFrame(index=ticker)
    df["Link"] = links
    df["GICS"] = sectors
    df["Text"] = ""
    
    #loop through all companies and get description
    for i in range(0,len(df)):
        print(df["Link"].iloc[i])
        page = requests.get(df["Link"].iloc[i])
        soup = bs.BeautifulSoup(page.content, 'html.parser')
       
        ps = soup.find_all("p")
        ps = [p.text for p in ps]
        txtps = " ".join(ps)
        txtps = re.sub("\[[0-9]+\]","",txtps) #remove supercript references
        
        txtps = stemmer.stem(txtps) #stem
        
        df["Text"].iloc[i] = txtps
        
        
        
    df.to_excel("SP100Wiki.xlsx")
    
else:
    df = pd.read_excel("SP100Wiki.xlsx")
    
    
    
###
#run analysis
###


tfidfvect = TfidfVectorizer(stop_words="english")
vects = tfidfvect.fit_transform(df["Text"])
scores = pd.DataFrame(cosine_similarity(vects))
scores.index = df.index
scores.columns = df.index
# plt.figure(figsize=(20,20))
# sns.heatmap(scores,annot=False,cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),cbar=False)

tk = ["AAPL","XOM","JPM"]

fig,ax = plt.subplots(1,len(tk),figsize=(17,6))
fig.suptitle("Top 5 Company Cosine Similarity Using Company Wikipedia Page")
i = 0
for t in tk:
    sc = np.sort(scores[t])[::-1][1:6]
    ind = scores.index[np.argsort(scores[t])[::-1]][1:6]

    sns.barplot(y=ind,x=sc,color="b",ax=ax[i])
    ax[i].set_title(t)
    ax[i].set_xlabel("Cosine Similarity")
    ax[i].set_ylabel("Ticker")
    i += 1


