The project largely uses publically available data from the Fama French Library.

## Style Analysis

A simple way to analyze factor exposure for a manager is using Sharpe Style analysis. In this process, we run a constrained linear regression over sliding windows to get the approximate style allocations for a return stream. 

This is done by solving the following optimization problem to minimize tracking error:

$$Min_{w'}  := {w'}^T\Sigma w'$$

$$s.t. w' + b \geq 0  \text{    Long only replication}$$

$${w}^T1 = 1  \text{    Fully invested}$$

$$\text{Where w' is the active portfolio weight (w-b), b is the benchmark weight, } \Sigma \text{ is the covariance matrix.}$$

As can be seen below for decomposition of the the cap-weighted return for the Oil industry, the return has largely been a reflection of beta to the overall stock market, which intuitively makes sense since there is economic linkage between both. We also note that post the 2014-2015 oil price crash, oil companies have had larger exposure to the value factor, largely due to the compression in their P/E as investors shunned the industry.

It is important to use relevant factor definitions or else the results may be difficult to interpret.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/styleoil.png)


## Portfolio Weighting Methodologies

There are many alternative methodologies to the Cap-weighted portfolio. Below we show a few iterations of Global Minimum Variance (GMV) portfolios. Since portfolio optimizations can be sensitive to input data. When building the covariance matrix for asset returns using a large number of assets, it is usually a good idea to provide some structure to the covariance matrix to improve upon the robustness of the results and reduce sensitivity. In the below we shrink the sample covariance matrix to a constant correlation form to balance the limitations of small sample size. Alternatively, more complex shrinking targets can be used such as structured statistical risk models (PCA) or fundamanetal risk models. It can be additionally prudent to add relevant constraints to industries or single assets to prevent optimizers from finding edge case solutions.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/ports.png)

## Risk Contributions

Investors in cap-weighted (CW) benchmarks often think they hold diversified portfolios. However this is often not the case. CW benchmarks can often become concentrated in certain sectors and industries as investors chase returns. One way of understanding this concentration risk is through analysis of Effective Number of Constituents (ENC) or Effective Number of Correlated Bets (ENCB). In a well diversified portfolio we would expect these numbers to remain high. 

In the case of 49 industry Fama-French CW benchmarks, we can see that even though there are 49 "assets" to choose from, the ENB/ENCB remain about half of that for the CW portfolio, signalling that this portfolio may not be as diversified as we originally thought. We can use a Risk Parity (RP) approach to building portfolios that maximizes the ENCB by targeting portfolio weights that keep risk contributions relatively equal for each asset. We should also note that the GMV portfolio has an even smaller ENB/ENCB as the optimizer crowds into low volatility areas of the market (another reason it is prudent to enforce constraints in portfolio optimization).

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/enbencb.png)

Below I show the top 10 risk contributions at the end of 2018 for the CW and EW portfolios. Of note, the CW portfolio has contribution from software names of 20% to the total risk of the portfolio. This should come as no surprise to U.S. investors as the late 2010's have seen the market-caps of the large U.S. tech names increase significantly. However, it is a reminder that CW portfolios may be more concentrated than naively thought.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/rcont.png)
