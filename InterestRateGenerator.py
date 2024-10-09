import numpy as np
from abc import ABC, abstractmethod

class InterestRateGenerator(ABC):
    """let's get pricing! 
    subclasses include:
    - RandomWalkPriceGenerator
    - GBMPriceGenerator
    - HestonPriceGenerator
    """
    @abstractmethod
    def generate_prices(self):
        pass

class VasicekPriceGenerator(InterestRateGenerator):
    def __init__(self, r0, kappa, theta, sigma, T, N):
        """
        Simulate interest rates using the Vasicek model.
        
        Parameters:
        r0 (float): Initial interest rate.
        kappa (float): Speed of mean reversion.
        theta (float): Long-term mean interest rate.
        sigma (float): Volatility of interest rate.
        T (float): Total time period.
        N (int): Number of time steps.
        """
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.T = T
        self.N = N

    def generate_rates(self):
        dt = self.T / self.N
        rates = np.zeros(self.N)
        rates[0] = self.r0
        
        for t in range(1, self.N):
            dr = self.kappa * (self.theta - rates[t-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates[t] = rates[t-1] + dr
        
        return rates
    
