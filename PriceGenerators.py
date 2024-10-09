import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class PriceGenerator(ABC):
    """let's get pricing! 
    subclasses include:
    - RandomWalkPriceGenerator
    - GBMPriceGenerator
    - HestonPriceGenerator
    """
    @abstractmethod
    def generate_prices(self):
        pass

class RandomWalkGenerator(PriceGenerator):
    def __init__(self,S0, mu , sigma,T, N):
        """
        Simple way of looking at prices of an asset:

        in: (floats)
        initials:   S0 (price)
        average:    mu (daily returns)
                    sigma (volatility)
        time:       T (total time) , N: (no. time steps)

        out: array price
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N = N
    
    def generate_prices(self):
        dt = self.T / self.N
        returns = np.random.normal(loc=self.mu, scale=self.sigma*np.sqrt(dt), size=self.N)
        S = self.S0 * np.cumprod(returns)
        return S

class GBMPriceGenerator(PriceGenerator):

    def __init__(self, S0, mu, sigma, T, N):
        """Simulating asset prices with Brownian Motion:
            in: (floats)
            initials:   S0 (price)
            average:    mu (daily returns)
                        sigma (volatility)
            time:       T (total time) , N: (no. time steps)
            out: array price
     """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N = N

    def generate_prices(self):
        dt = self.T / self.N
        t = np.linspace(0, self.T, self.N)
        W = np.random.standard_normal(size=self.N)
        W = np.cumsum(W) * np.sqrt(dt)
        X = (self.mu - 0.5 * self.sigma**2) * t + self.sigma * W
        S = self.S0 * np.exp(X)
        return S






class HestonPriceGenerator(PriceGenerator):
    def __init__(self, S0, V0, mu, kappa, theta, xi, rho, T, N):
        """GBM pricing with stochastic volatility!

        dS = mu Sdt + sqrt(V) SdW_1

        dV = kappa(theta - V) + xi sqrt(V)dW_2
    
        in: (floats)
            initials:   S0 (price) , V0 (variance = volatility^2)
            averages:   mu (log returns), theta (variance= volatility^2)
            response:   xi (volatility on volatility), kappa (rate of volatility mean reversion)
            corelation: rho (correlation between Wiener processes for price and volatility)
            time: T (total simulation time) , N (no. timseteps)

        out:
            array: price
            """
        self.S0 = S0
        self.V0 = V0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.T = T
        self.N = N

    def generate_prices(self):
        dt = self.T / self.N
        S = np.zeros(self.N)
        V = np.zeros(self.N)
        S[0] = self.S0
        V[0] = self.V0
        
        for t in range(1, self.N):
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = np.random.normal(0, np.sqrt(dt))
            dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * dW2
            
            V[t] = V[t-1] + self.kappa * (self.theta - V[t-1]) * dt + self.xi * np.sqrt(V[t-1]) * dW2
            V[t] = max(V[t], 0)
            S[t] = S[t-1] * np.exp((self.mu - 0.5 * V[t-1]) * dt + np.sqrt(V[t-1]) * dW1)
        
        return S
    


class CorrelatedRandomWalkGenerator(PriceGenerator):
    def __init__(self, S0s, mus, sigmas, T, N, correlation_matrix):
        """
        Simulate correlated (or anticorrelated) asset prices:

        S0s: Initial prices of the assets (list)
        mus: Expected daily returns of the assets (list)
        sigmas: Volatility of the assets (list)
        T: Total time (float)
        N: Number of time steps (int)
        correlation_matrix: Correlation matrix between the assets (numpy array)
        """
        self.S0s = np.array(S0s)
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.T = T
        self.N = N
        self.correlation_matrix = correlation_matrix
    
    def generate_prices(self):
        dt = self.T / self.N
        
        # Generate uncorrelated returns
        uncorrelated_returns = np.random.normal(
            loc=self.mus * dt,
            scale=self.sigmas * np.sqrt(dt),
            size=(self.N, len(self.S0s))
        )
        
        # Apply Cholesky decomposition to create correlated returns
        L = np.linalg.cholesky(self.correlation_matrix)
        correlated_returns = uncorrelated_returns @ L.T
        
        # Convert correlated returns into price paths
        prices = np.zeros_like(correlated_returns)
        prices[0] = self.S0s
        
        for t in range(1, self.N):
            prices[t] = prices[t - 1] * (1 + correlated_returns[t])
        
        return prices

class CandlestickSimulator(PriceGenerator):
    def __init__(self, S0, mu, sigma, T, N):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N = N

    def generate_prices(self):
        dt = self.T / self.N
        returns = np.random.normal(loc=self.mu * dt, scale=self.sigma * np.sqrt(dt), size=self.N)
        prices = self.S0 * np.exp(np.cumsum(returns))
        
        opens = np.zeros(self.N)
        highs = np.zeros(self.N)
        lows = np.zeros(self.N)
        closes = np.zeros(self.N)
        
        # Initialize the first row
        opens[0] = self.S0
        closes[0] = prices[0]
        highs[0] = max(opens[0], closes[0])  # Since we don't have prior data, high can be open or close
        lows[0] = min(opens[0], closes[0])   # Same for low
        
        for t in range(1, self.N):
            opens[t] = closes[t-1]
            rand_high = np.random.uniform(0, 1)
            rand_low = np.random.uniform(0, 1)
            high = opens[t] * (1 + rand_high * self.sigma * np.sqrt(dt))
            low = opens[t] * (1 - rand_low * self.sigma * np.sqrt(dt))
            closes[t] = prices[t]
            highs[t] = max(high, closes[t])
            lows[t] = min(low, closes[t])
        
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=self.N, freq='D')
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes
        }, index=dates)
        
        return df