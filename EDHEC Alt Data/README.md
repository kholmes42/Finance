## Alternative Data

Typical financial investment analysis uses fundamental ratios found from company financial statements or technical analysis using prices and volume. As markets become more efficient, more focus is turning to real time alternative datasets. For example, investment analysts may look at location data or trip data to better understand consumer consumption in real time. In the below we take a look at Uber trip data in NYC from July 2014 to see if there are any trends in the data which could then be used to look at consumer behaviour. Unsurprisingly, we find NYC Uber trips differ at statistically significant levels between weekends and weekdays. Appearing to show an increase during morning and evening commutes on weekdays and an increase very late at night on weekends as people return home from parties. Additional analysis can include locations that are frequented most and examining if these are changing over time.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/uberdata.png)

## Company Similarity Using Wikipedia

Companies are usually categorized by GICS sector classification into 11 sectors. We can use alternative methods to try and examine if companies are more similar to others as well. In the below example I use TF-IDF on the SP100 company's Wikipedia pages to vectorize their descriptions. Subsequently I use pairwise cosine similarity to examine the top 5 similar textual descriptions of Apple, Exxon, and JP Morgan. Unsurprisingly the top five companies that are found by the algorithm are fairly intuitive, largely belonging to the same industries.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/textsimilarity.png)
