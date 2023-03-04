The project largely uses publically available data from the Fama French Library.

## Style Analysis

A simple way to analyze factor exposure for a manager is using Sharpe Style analysis. In this process, we run a constrained linear regression over sliding windows to get the approximate style allocations for a return stream. 

This is done by solving the following optimization problem to minimize tracking error:

$$Min_{w'}  := {w'}^T\Sigma w'$$

$$s.t. w' + b \geq 0  \text{    Long only replication}$$

$${w'}^T1 = 1  \text{    Fully invested}$$

$$\text{Where w' is the active portfolio weight, b is the benchmark weight, } \Sigma \text{ is the covariance matrix.}$$

As can be seen below for decomposition of the the cap-weighted return for the Oil industry, the return has largely been a reflection of beta to the overall stock market, which intuitively makes sense since there is economic linkage between both. We also note that post the 2014-2015 oil price crash, oil companies have had larger exposure to the value factor, largely due to the compression in their P/E as investors shunned the industry.

It is important to use relevant factor definitions or else the results may be difficult to interpret.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/styleoil.png)


## Portfolio Weighting Methodologies

There are many alternative methodologies to the Cap-weighted portfolio. Below we show a few iterations of Global Minimum Variance (GMV) portfolios. Since portfolio optimizations can be sensitive to input data. When building the covariance matrix for asset returns using a large number of assets, it is usually a good idea to provide some structure to the covariance matrix to improve upon the robustness of the results and reduce sensitivity. In the below we shrink the sample covariance matrix to a constant correlation form to balance the limitations of small sample size. Alternatively, more complex shrinking targets can be used such as structured statistical risk models (PCA) or fundamanetal risk models. It can be additionally prudent to add relevant constraints to industries or single assets to prevent optimizers from finding edge case solutions.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/ports.png)
