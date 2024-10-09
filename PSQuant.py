"""My own quant library!!!"""

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm


### DATA FETCHING
def FetchData(ticker, start_date, end_date):
    """Fetches data from yfinance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

### TECHNICAL INDICATORS
def SMA(prices, window):
    """Simple Moving Average over window"""
    return prices.rolling(window=window).mean()

def LogReturns(prices):
    """log(price_n / price(n-1))"""
    return prices.log().diff()

# def LogReturns_mean(prices, window):
#     return prices.rolling(window=window).log().diff().mean()

# def LogReturns_std(prices, window):
#     return prices.rolling(window=window).log().diff().std()

# Assuming these functions are in PSQ
def LogReturns_std(prices, window):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).std()

def LogReturns_mean(prices, window):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).mean()


def BollingerBands(prices, window, num_std_dev):
    """calculates bollinger bands
    out: lower_band, sma, upper_band"""
    sma = SMA(prices, window)
    std_dev = prices.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return lower_band, sma, upper_band

def CalculateReturns(prices):
    """returns returns (percentages)"""
    return prices.pct_change().dropna()

def CalculateVolatility(returns):
    return returns.std()

def PortfolioReturns(weights, returns):
    """ in:  weights & returns
        out: portfolio returns
    """
    return (weights * returns).sum(axis=1)

def Sharpe(returns, risk_free_rate=0.01, period = 365):
    excess_returns = returns.mean() - risk_free_rate/period
    return excess_returns / returns.std()

def BacktestStrategy(data, signal, initial_cash=10000):
    cash = initial_cash
    position = 0
    equity = []
    
    for i in range(len(data)):
        if signal.iloc[i] == 1 and position ==0:  # Buy
            position = cash / data['Close'].iloc[i]
            cash -= position*data['Close'].iloc[i]
        elif signal.iloc[i] == -1 and position >0:  # Sell
            cash += position * data['Close'].iloc[i]
            position = 0
        
        equity.append(cash + position * data['Close'].iloc[i] if position > 0 else cash)
    
    return pd.Series(equity, index=data.index)

def ZScore(series):
    return (series - series.mean()) / series.std()

def MeanReversionStrat(prices, window, z_threshold):
    z_scores = ZScore(prices.rolling(window).mean())
    buy_signals = z_scores < -z_threshold
    sell_signals = z_scores > z_threshold
    return buy_signals.astype(int) - sell_signals.astype(int)


def ValueAtRisk(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) of a portfolio.

    Parameters:
    - returns: A pandas Series or numpy array of returns.
    - confidence_level: The confidence level for VaR calculation (default is 95%).

    Returns:
    - var: The Value at Risk (VaR) at the specified confidence level.
    """
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def ExpectedShortFall(returns, confidence_level=0.95):
    """
    Calculate the Expected Shortfall (ES) of a portfolio.

    Parameters:
    - returns: A pandas Series or numpy array of returns.
    - confidence_level: The confidence level for ES calculation (default is 95%).

    Returns:
    - es: The Expected Shortfall (ES) at the specified confidence level.
    """
    var = ValueAtRisk(returns, confidence_level)
    es = returns[returns <= var].mean()
    return es



def LinRegHedgeRatio(y, x):
    """
    Calculate the hedge ratio using linear regression.

    Parameters:
    - y: Dependent variable (e.g., price of asset to hedge).
    - x: Independent variable (e.g., price of asset used for hedging).

    Returns:
    - hedge_ratio: The calculated hedge ratio.
    """
    x = sm.add_constant(x)  # Adds a constant term to the model
    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[1]  # Slope of the regression line
    return hedge_ratio

def PairsTradingStrat(spread, threshold=1.0):
    """
    Generate trading signals for pairs trading based on z-score of spread.

    Parameters:
    - spread: The price spread between the two assets.
    - threshold: The z-score threshold for generating signals.

    Returns:
    - signals: A pandas Series of trading signals (1 = buy, -1 = sell).
    """
    z_score = (spread - spread.mean()) / spread.std()
    signals = np.where(z_score > threshold, -1, np.where(z_score < -threshold, 1, 0))
    return pd.Series(signals, index=spread.index)


# portfolio optimisation

import cvxpy as cp

def mean_variance_optimization(returns, target_return):
    """
    Perform mean-variance optimization to find the optimal portfolio weights.

    Parameters:
    - returns: A pandas DataFrame of asset returns.
    - target_return: The target portfolio return.

    Returns:
    - weights: The optimized portfolio weights.
    """
    n_assets = returns.shape[1]
    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    
    weights = cp.Variable(n_assets)
    portfolio_return = mean_returns @ weights
    portfolio_volatility = cp.quad_form(weights, cov_matrix)
    
    objective = cp.Minimize(portfolio_volatility)
    constraints = [cp.sum(weights) == 1, portfolio_return >= target_return, weights >= 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return weights.value


# def CalculateDrawdowns(equity_series):
#     # Calculate the running maximum
#     running_max = equity_series.cummax()

#     # Calculate drawdown as the percentage drop from the running maximum
#     drawdowns = (equity_series - running_max) / running_max

#     return drawdowns

def CalculateDrawdowns(equity_series):
    # Calculate the running maximum
    running_max = equity_series.cummax()

    # Calculate drawdown as the percentage drop from the running maximum
    drawdowns = (equity_series - running_max) / running_max

    return drawdowns * 100  # Convert to percentage









