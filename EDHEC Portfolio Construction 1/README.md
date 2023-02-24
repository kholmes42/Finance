The project largely uses publically available data from the Fama French Library.

## Efficient Frontier

We begin with the simple depiction of the efficient frontier for a mean variance investor. Under EMH the maximizing sharpe ratio portfolio is the market portfolio as indicated by the red dot. However, optimizers are very sensitive to risk and return inputs. Therefore it is difficult to say with certainty what this portfolio comprises of. Given that volatility estimates tend to be relatively more stable, an alternative for an investor is the Global Minimum Variance (GMV orange dot) portfolio which mimimzes risk as measured by variance of returns. We also indicate a simple equal weighted portfolios which plots close to the GMV without having to rely on any input risk.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/Frontier.png)


## Limits of Diversification

We next examine the premise of diversification. It generally works well in the long run, leading to smoother investment returns, however in scenarios where systemic risk rises, correlations tend to increase. This means correlations can fail in tail events when investors would benefit from it the most.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/RollingCorrels.png)

## CPPI

To counteract the limits of diversification, we investigate the rules based investment strategy called Constant Portfolio Protection Insurance (CPPI) in two configurations (naive and Max DD). CPPI stratgies dynamically rebalance into riskless assets from risky assets in order to maintain a mimimum acceptable portfolio value. However, investors need to be aware that they can be susceptible to gap/crash risk where the market can move very quickly through the floor. If many investors are following a similar strategy this can result in added selling pressure to the risky asset as many investors try to flee at once causing liquidity issues. Ultimately this means that investors may be stuck in a position out of the market in the case of a quick rebound.

![plot](https://github.com/kholmes42/Finance/blob/main/imgs/CPPI.png)
