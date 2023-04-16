# -*- coding: utf-8 -*-
"""
Created on Sun Apr  16 20:03:15 2023
EDHEC Week 3 Company Filings
@author: kholm
dataset: EDGAR Filings 13F
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sec_edgar_downloader import Downloader
import bs4 as bs
import re
from collections import defaultdict

tickers = ['0001423053'] #use SEC CIK number if duplicates exist
filetype = "13F-HR"

read_new = False

###
#download files and save to disk
###
if read_new == True:
    dl = Downloader("Filings")
    for tk in tickers:
        print(tk)
        dl.get(filetype, tk, after="2022-01-01", before="2023-04-01")
    
 
    
 
    
def parse_hold_section13F(doc_dict):
    """
    This function parses the holdings section of a company 13F
    """

    doc_year_dict = defaultdict(lambda: {})
    
    for tk,dc in doc_dict.items():    
        #manipulate file string                

        for d in list(dc):
            
            doc = bs.BeautifulSoup(d, "lxml")
            
            l = re.findall("CONFORMED PERIOD OF REPORT:[\s]*[0-9]{8}",doc.text)
            print(l)
            
            
            tbl = doc.find_all('informationtable')[0]
            
            inftables = tbl.find_all("infotable")
   
            #print([tag.name for tag in inftable.find_all()])
           
            data = []
            for row in inftables:
                nm = row.find('nameofissuer')
                title = row.find('titleofclass')
                val = row.find('value')
                pc = row.find("putcall")
                shr = row.find("shrsorprnamt")
                typ = row.find('sshprnamttype')
                
                if pc is None:
                    pc = "N/A"
                else:
                    pc = pc.get_text()
                
                data.append([nm.get_text(),title.get_text(),val.get_text(),pc,re.sub("[^0-9]","",shr.get_text()),typ.get_text()])
            
            dt = pd.to_datetime(l[0][-8:])
            
            # Converting the list into dataframe
            df = pd.DataFrame(data, columns=['Name',"Title","Value","PC","shares","sharetype"])
            df["Value"] = df["Value"].astype(float)
            df["shares"] = df["shares"].astype(float)
            print(df.head())
            doc_year_dict[tk][dt] = df
            
    return doc_year_dict
 
    
    
def read_files_from_disk(tickers,doctype="10-K"):
    """
    This function reads files from disk and uses a custom helper function to search
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
     
    if doctype == "10-K:":
        
        lookup = parse_risk_section10K(doc_dict)
    else:
        lookup = parse_hold_section13F(doc_dict)
    
    return lookup



###
#read in files from drive
###

fileout = read_files_from_disk(tickers,filetype)


df = fileout[tickers[0]][pd.to_datetime("2022-12-31")]
df = df.loc[df["PC"] == "N/A"][["Name","Value"]].groupby(["Name"]).sum()

df /= df["Value"].sum()

df.sort_values(by=["Value"],inplace=True,ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x=df.iloc[:10]["Value"]*100,y=df.iloc[:10].index,color="b")
plt.title("Top 10 Weights for Citadel Advisors LLC as of 12-31-2022")
plt.xlabel("Weight (%)")
plt.ylabel("Name")