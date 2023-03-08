## Factor Models

OLS Regression based factor models can be sensitive to leverage points (outliers) and collinearity in regressors. Collinearity is often found in financial data since there are many common influences across financial markets. In order to provide more robust beta estimations for factor models, we can add regularization terms to shrink the impact of these effects. Below we show the betas from a simple OLS factor model along with a LASSO factor model which encourages sparsity among coefficients. The LASSO regression show that many of the asset returns along the y axis are more clearly explained through the World Equity return stream. The exception is Corporate Bonds which is a blend between tresuries and equity risk which makes intuitive sense. The sparsity from the LASSO rgressors also encourages increased interpretability.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/factcoef.png)

