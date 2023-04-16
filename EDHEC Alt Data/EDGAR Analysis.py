# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:03:15 2023
EDHEC Week 3 Company Filings
@author: kholm
dataset: EDGAR Filings 10-K
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sec_edgar_downloader import Downloader
import bs4 as bs
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer 
from collections import defaultdict
from nltk import word_tokenize          
from nltk.corpus import stopwords

tickers = ['XOM'] #use SEC CIK number if duplicates exist
filetype = "10-K"

read_new = False

###
#download files and save to disk
###
if read_new == True:
    dl = Downloader("Filings")
    for tk in tickers:
        print(tk)
        dl.get(filetype, tk, after="2023-01-01", before="2023-04-01")
    
 
    
#refer to Sklearn documentation for credit   
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]   

    
   
def parse_risk_section10K(doc_dict):
    """
    This function parses the Risk Factors section of a company 10k
    """

    doc_year_dict = defaultdict(lambda: {})
    
    for tk,dc in doc_dict.items():    
        #manipulate file string                
        for d in list(dc):
        
            doc = bs.BeautifulSoup(d, "lxml")
            l = re.findall("FILED AS OF DATE:[\s]*[0-9]{8}",doc.text)
            print(l)
           
            beg = list(re.finditer("Item[\s]*1A\.",doc.text,re.IGNORECASE))
            fin = list(re.finditer("Item[\s]*1B\.",doc.text,re.IGNORECASE))
  
            try:
                yr = l[0][-8:][:4]  
     
                risksection = doc.text[beg[1].end():fin[1].start()].replace("PART I"," ").replace("Item 1A"," ")
                risksection = re.sub("\d+", " ",risksection)
                risksection = re.sub("\n", " ",risksection)
                risksection = re.sub("•", " ",risksection)
      
                  
                doc_year_dict[tk][yr] = risksection
            except: 
                print(yr)
      
    return doc_year_dict
 


def read_files_from_disk(tickers,doctype="10-K"):
    """
    This function reads files from disk and uses a custom helper function to search for risk factor section
    """

    doc_dict = defaultdict(lambda: [])
    filetext = []
    for tk in tickers:
        
        pth = os.path.join(r"Filings\sec-edgar-filings",tk , doctype)
        print(tk)
        for filename in os.listdir(pth):
            f = os.path.join(pth, filename,"full-submission.txt")
            # checking if it is a file
            print(f)
            if os.path.isfile(f):
                with open(f, 'r') as file:
                    data = file.read()
                    doc_dict[tk].append(data)
                    filetext.append(data)
     

    lookup = parse_risk_section10K(doc_dict)
    
    return lookup

   

###
#read in files from drive
###

fileout = read_files_from_disk(tickers,filetype)

for tk in fileout.keys():

    df = pd.DataFrame(index=fileout[tk].keys(),data=fileout[tk].values())
    df.columns = ["Risk Factors"]


df.sort_index(inplace=True)




###
#vectorize word text
###
stops = set(stopwords.words('english'))
stops2 = ["u","-k","|","d","'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would',
          "eu","’","'",'“', '”',"'d","●",'and/or',"mr.","inc.",
          ',', '.','inc', 'mr',';',"may"]

stops = stops.union(set(stops2))
vectorizer = TfidfVectorizer(stop_words=stops,max_features=5000,tokenizer=LemmaTokenizer())

X = vectorizer.fit_transform(df["Risk Factors"])

###
#analyze similarity between sections for each ticker year over year
###
sim = []
for i in range(1,X.shape[0]):
    sim.append(np.abs(cosine_similarity(X[i,:],X[i-1,:]))[0][0])
    
    
plt.figure()
sns.barplot(y=sim,x=list(df.index[1:]),color="b")
plt.title(tickers[0] + " " + filetype + " YoY Cosine Similarity of Risk Factor Section")
plt.xlabel("Year vs Prior Year (Filed by Early Feb of Listed Year)")
plt.ylabel("Cosine Similarity")