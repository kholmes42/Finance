## Factor Models

OLS Regression based factor models can be sensitive to leverage points (outliers) and collinearity in regressors. Collinearity is often found in financial data since there are many common influences across financial markets. In order to provide more robust beta estimations for factor models, we can add regularization terms to shrink the impact of these effects. Below we show the betas from a simple OLS factor model along with a LASSO factor model which encourages sparsity among coefficients. The LASSO regression show that many of the asset returns along the y axis are more clearly explained through the World Equity return stream. The exception is Corporate Bonds which is a blend between tresuries and equity risk which makes intuitive sense. The sparsity from the LASSO rgressors also encourages increased interpretability.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/factcoef.png)

In order to provide structure to the covariance matrix we can also apply a factor model. This decomposes the asset covariance matrix into:

$$ \Sigma_Z = \beta \Sigma_F \beta^T + \Sigma_{ee} $$

$$\text{Where } \beta \text{ is the asset factor exposures from linear regression (n asset by k factor), } $$

$$ \Sigma_F \text{ is the factor covariance matrix, and } \Sigma_{ee} \text{ is a diagonal matrix of specific risks assumed to be uncorrelated}$$

It also allows the decomposition of portfolio risk into specific risk and factor risk.


![plot](https://github.com/kholmes42/Finance/blob/main/imgs/riskbreakdown.png)

## Graphical Network Visualization of Covariance Matrix

We can build a graphical representation of the covariance matrix to get an interpretable visualization of our assets. This is done in 4 steps.

1. Use Graphical LASSO to build a sparse representation of the percision matrix (inverse covariance matrix)
2. Cluster in the covariance matrix space to get hidden clusters of assets
3. Use a dimensionality reduction technique (ex. MDS) to get 2D embedding of covariance matrix
4. Use step 1 to get edges, step 2 to get node colors, step 3 to get node locations for 2D visualization 

Using our algorithm we get 10 clusters using returns from 2010-2018. We try to interpret the clusters and provide meaningful names to them. Intuitively, similar industries group together and are more isolated from others. For example, Staples (orange) and Defensives (red) are less connected to other industries because they are less economically sensititve in nature. Additionally, we notice Gold (pink) is very isolated from all other assets as it is often one of the least correlated industries to the broader market. We also note that Services (turqoise) with a large financial representation and Industrials (purple) are highly connected as they are generally more cyclical areas of the market.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/graphicalnetwork.png)
