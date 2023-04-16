## Alternative Data

Typical financial investment analysis uses fundamental ratios found from company financial statements or technical analysis using prices and volume. As markets become more efficient, more focus is turning to real time alternative datasets. For example, investment analysts may look at location data or trip data to better understand consumer consumption in real time. In the below we take a look at Uber trip data in NYC from July 2014 to see if there are any trends in the data which could then be used to look at consumer behaviour. Unsurprisingly, we find NYC Uber trips differ at statistically significant levels between weekends and weekdays. Appearing to show an increase during morning and evening commutes on weekdays and an increase very late at night on weekends as people return home from parties. Additional analysis can include locations that are frequented most and examining if these are changing over time.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/uberdata.png)

## Company Similarity Using Wikipedia

Companies are usually categorized by GICS sector classification into 11 sectors. We can use alternative methods to try and examine if companies are more similar to others as well. In the below example I use TF-IDF on the SP100 company's Wikipedia pages to vectorize their descriptions. Subsequently I use pairwise cosine similarity to examine the top 5 similar textual descriptions of Apple, Exxon, and JP Morgan. Unsurprisingly the top five companies that are found by the algorithm are fairly intuitive, largely belonging to the same industries.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/textsimilarity.png)


## Textual Risk Section Analysis of Company 10-Ks

We can pull company filings from the EDGAR (SEC) database for NLP analysis. We will try to analyze how "cookie cutter" the Risk Factor Item 1A sections are on a year over year basis. If we see high similarity between year over year it means the section is largely a copy paste job from the previous year. Drops in similarity could imply new risks are being added to the watch by management. This process involves using an API to download company filings, using regex to find relevant sections for analysis, using a count vectorizer to vectorize the text sections of the filing, and then finally comparing the cosine similarity of the vectors on a year over year basis.

The below example uses Exxon Mobil (XOM) for the analysis with 10-K filings from 2013-2022. Generally, we see pretty similar Risk Factor sections, with the only noticeable changes occuring in 2021 and 2022. Upon examination of the documents, this is because the first reference to the COVID pandemic occurs in the 2020 filing (filed in 2021) and a larger emphasis on climate risks in the 2021 filing (filed in 2022). Both are key risks that the company faced, first due to the large economic demand destruction that occured during the pandemic, with XOM being sensitve to this as an energy producer. Second, with the pressure that society is placing on the energy sector to transition to cleaner forms of energy and lower emissions.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/10kxom.png)
