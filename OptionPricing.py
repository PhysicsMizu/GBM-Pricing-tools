import math
from scipy.stats import norm
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class OptionPricing(ABC):
    """let's get pricing! 
    subclasses include:
    - RandomWalkPriceGenerator
    - GBMPriceGenerator
    - HestonPriceGenerator
    """
    @abstractmethod
    def price(self):
        pass
    @abstractmethod
    def delta(self):
        pass
    @abstractmethod
    def gamma(self):
        pass
    @abstractmethod
    def vega(self):
        pass
    @abstractmethod
    def theta(self):
        pass
    @abstractmethod
    def rho(self):
        pass




class BlackScholes(OptionPricing):
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        Initialize the Black-Scholes model parameters.
        
        :param S: Current stock price
        :param K: Option strike price
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate (annual)
        :param sigma: Volatility of the underlying asset (annual)
        :param option_type: 'call' for a call option, 'put' for a put option
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        
        # Calculate d1 and d2

        # cdf of standard normal distribution (risk-adjusted probability that the option will be exercised)
        self.d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))
        # cdf of standard normal distribution (probability of receiving the stock at expiration of the option)
        self.d2 = self.d1 - self.sigma * math.sqrt(self.T)
    
    def price(self):
        """
        Calculate the Black-Scholes option price.
        
        :return: The price of the call or put option
        """
        if self.option_type == 'call':

            # price = S*N(d1) - PresentValue(K)*N(d2)
            price = (self.S * norm.cdf(self.d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2))
        elif self.option_type == 'put':
            # price = PresentValue(K)*N(-d2) - S*N(-d1)
            price = (self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1))
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")
        return price
    
    def delta(self):
        """
        Calculate the Delta of the option.
        Measures the sensitivity of the option price to changes in the underlying asset price.
        dprice/dS
        :return: Delta value
        """
        if self.option_type == 'call':
            return norm.cdf(self.d1)
        elif self.option_type == 'put':
            return norm.cdf(self.d1) - 1
    
    def gamma(self):
        """
        Calculate the Gamma of the option.
        Measures the sensitivity of the option price to changes in delta.
        dprice/ddelta = d^2 price / d^2 S
  
        :return: Gamma value
        """
        return norm.pdf(self.d1) / (self.S * self.sigma * math.sqrt(self.T))
    
    def vega(self):
        """
        Calculate the Vega of the option.
        Measures the sensitivity of the option price to changes in volatility.
        d price / d sigma
        
        :return: Vega value
        """
        return self.S * norm.pdf(self.d1) * math.sqrt(self.T)
    
    def theta(self):
        """
        Calculate the Theta of the option.
        Measures the sensitivity of the option price to the passage of time (time decay).
        d price/ dt
        :return: Theta value
        """
        if self.option_type == 'call':
            term1 = -self.S * norm.pdf(self.d1) * self.sigma / (2 * math.sqrt(self.T))
            term2 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2)
            return (term1 - term2) / 365
        elif self.option_type == 'put':
            term1 = -self.S * norm.pdf(self.d1) * self.sigma / (2 * math.sqrt(self.T))
            term2 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return (term1 + term2) / 365
    
    def rho(self):
        """
        Calculate the Rho of the option.
        Measures the sensitivity of the option price to changes in the risk-free interest rate.
        dprice/dr
        :return: Rho value
        """
        if self.option_type == 'call':
            return self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        elif self.option_type == 'put':
            return -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
        

class MonteCarloOption(OptionPricing):

    def __init__(self, S, K, T, r, sigma, option_type='call', samplesize = 1, epsilon=1e-1):
        """
        Initialize the Black-Scholes model parameters.
        
        :param S: Current stock price
        :param K: Option strike price
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate (annual)
        :param sigma: Volatility of the underlying asset (annual)
        :param option_type: 'call' for a call option, 'put' for a put option
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.numsteps = int(self.T*365) # 1 step a day
        self.samplesize = samplesize
        self.epsilon = epsilon
        self.epsilon_s = epsilon*self.S
        self.epsilon_sigma = epsilon*self.sigma
        self.epsilon_t = epsilon*self.T
        self.epsilon_r = epsilon*np.abs(r)




    def simulate_GBM(self):
        #simulates geometric brownian motion efficiently!
        dt = self.T /self.numsteps
        t = np.linspace(0, self.T, self.numsteps)
        W = np.random.standard_normal(size=self.numsteps)
        W = np.cumsum(W) * np.sqrt(dt)
        X = (self.r - 0.5 * self.sigma**2) * t + self.sigma * W
        S = self.S * np.exp(X)
        return S[-1]
    
    def price(self, S=None, sigma=None, T=None, r=None, return_price_std = False):
        if S is None: S = self.S
        if sigma is None: sigma = self.sigma
        if T is None:T = self.T
        if r is None:r = self.r
        
        S_samples = np.zeros(self.samplesize)
        for i in range(self.samplesize):
            self.S = S
            self.sigma = sigma
            self.T = T
            self.r = r
            S_samples[i] = self.simulate_GBM()

        if self.option_type == 'call':
            option_prices = np.maximum(S_samples - self.K, 0)
        elif self.option_type == 'put':
            option_prices = np.maximum(self.K - S_samples, 0)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        option_prices *= np.exp(-self.r * self.T)
        average_price = np.mean(option_prices)
        std_price = np.std(option_prices)

        if return_price_std: return average_price , std_price
        else: return average_price

    # def price(self, return_price_std = False):
    #     # S_sample = np.zeros(self.samplesize)
    #     # for i in range(self.samplesize):
    #     #     S_sample[i] = self.simulate_GBM()
        
    #     # #now, we can make an array of option prices
    #     # if self.option_type == 'Call':
    #     #     prices = np.maximum(S_sample - self.K, 0)
    #     # elif self.option_type == 'put':
    #     #     option_prices = np.maximum(self.K - S_sample, 0)
    #     # else:
    #     #     raise ValueError("Option type must be either 'call' or 'put'.")

    #     # # Discount to present value
    #     # option_prices *= np.exp(-self.r * self.T)

    #     # # Calculate the average and standard deviation of the option prices
    #     # average_price = np.mean(option_prices)
    #     # price_std_deviation = np.std(option_prices)

    #     # if return_price_std: return average_price, price_std_deviation
    #     # else: return average_price
    def delta(self):
        """
        Calculate the Delta of the option.
        Measures the sensitivity of the option price to changes in the underlying asset price.
        dprice/dS
        USES FINITE-ELEMENT DERIVATIVES
        :return: Delta value
        """
        original_S = self.S
        price_up = self.price(S=original_S + self.epsilon_s)
        price_down = self.price(S=original_S- self.epsilon_s)
        #S returns to original value
        self.S = original_S
        return (price_up - price_down) / (2 * self.epsilon_s)
    

        
    
    def gamma(self):
        """
        Calculate the Gamma of the option.
        Measures the sensitivity of the option price to changes in delta.
        dprice/ddelta = d^2 price / d^2 S
        USES FINITE-ELEMENT DERIVATIVES
        :return: Gamma value
        """
        original_S = self.S
        price_up = self.price(S=original_S + self.epsilon_s)
        price_down = self.price(S=original_S - self.epsilon_s)
        price_current = self.price(S=original_S)
        self.S = original_S
        return (price_up - 2 * price_current + price_down) / (self.epsilon_s ** 2)
    
    def vega(self):
        """
        Calculate the Vega of the option.
        Measures the sensitivity of the option price to changes in volatility.
        d price / d sigma
        USES FINITE-ELEMENT DERIVATIVES
        :return: Vega value
        """
        original_sigma = self.sigma
        price_up = self.price(sigma=original_sigma + self.epsilon_sigma)
        price_down = self.price(sigma=original_sigma- self.epsilon_sigma)
        self.sigma = original_sigma
        return (price_up - price_down) / (2 * self.epsilon_sigma)
  
    def theta(self):
        """
        Calculate the Theta of the option.
        Measures the sensitivity of the option price to the passage of time (time decay).
        d price/ dt
        USES FINITE-ELEMENT DERIVATIVES
        :return: Theta value
        """
 
        original_T = self.T
        price_up = self.price(T=original_T - self.epsilon_t)
        price_down = self.price(T=original_T + self.epsilon_t )
        self.T = original_T
        return (price_up - price_down) / (2*self.epsilon_t)
    
    def rho(self):
        """
        Calculate the Rho of the option.
        Measures the sensitivity of the option price to changes in the risk-free interest rate.
        dprice/dr
         USES FINITE-ELEMENT DERIVATIVES
        :return: Rho value
        """
        original_r = self.r
        price_up = self.price(r=original_r + self.epsilon_r)
        price_down = self.price(r=original_r - self.epsilon_r)
        self.r = original_r
        return (price_up - price_down) / (2 * self.epsilon_r)
 
    








class BinomialOptionPricing:
    def __init__(self, S0, K, T, r, sigma, option_type='call', epsilon = 1e-04):
        self.S0 = S0      # Initial stock price
        self.K = K        # Strike price
        self.T = T        # Time to maturity
        self.r = r        # Risk-free interest rate
        self.sigma = sigma # Volatility
        self.N = int(T*365*4)
        self.dt = self.T/self.N        # Number of steps to maturity
        self.option_type = option_type.lower()  # 'call' or 'put'

        # Small perturbation for numerical derivatives
        self.epsilon = epsilon  
        self.epsilon_s = epsilon*self.S0
        self.epsilon_sigma = epsilon*self.sigma
        self.epsilon_t = epsilon*self.T
        self.epsilon_r = epsilon*np.abs(r)
        
        # Calculating derived parameters
        self.u = np.exp(sigma * np.sqrt(self.dt)) # Up factor
        self.d = 1 / self.u                     # Down factor
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d) # Risk-neutral probability

    # def price(self):
    #     # Initialize asset prices at maturity
    #     ST = np.zeros(self.N + 1)
    #     for i in range(self.N + 1):
    #         ST[i] = self.S0 * (self.u ** (self.N - i)) * (self.d ** i)

    #     # Initialize option values at maturity
    #     if self.option_type == 'call':
    #         option_values = np.maximum(0, ST - self.K)
    #     elif self.option_type == 'put':
    #         option_values = np.maximum(0, self.K - ST)
    #     else:
    #         raise ValueError("Option type must be either 'call' or 'put'.")

    #     # Step backwards through the tree
    #     for j in range(self.N - 1, -1, -1):
    #         for i in range(j + 1):
    #             option_values[i] = np.exp(-self.r * self.dt) * (self.p * option_values[i] + (1 - self.p) * option_values[i + 1])

    #     # Option price at the root of the tree
    #     return option_values[0]
    




    def price(self, S0=None, sigma=None, T=None, r=None):
        if S0 is None:S0 = self.S0
        if sigma is None:sigma = self.sigma
        if T is None:T = self.T
        if r is None:r = self.r

        self.dt = T / self.N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)

        # Initialize asset prices at maturity
        ST = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            ST[i] = S0 * (self.u ** (self.N - i)) * (self.d ** i)

        # Initialize option values at maturity
        if self.option_type == 'call':
            option_values = np.maximum(0, ST - self.K)
        elif self.option_type == 'put':
            option_values = np.maximum(0, self.K - ST)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        # Step backwards through the tree
        for j in range(self.N - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-self.r * self.dt) * (self.p * option_values[i] + (1 - self.p) * option_values[i + 1])

        return option_values[0]

    def delta(self):
        """
        Calculate the Delta of the option.
        Measures the sensitivity of the option price to changes in the underlying asset price.
        dprice/dS
        USES FINITE-ELEMENT DERIVATIVES
        :return: Delta value
        """
        price_up = self.price(S0=self.S0 + self.epsilon_s)
        price_down = self.price(S0=self.S0 - self.epsilon_s)
        return (price_up - price_down) / (2 * self.epsilon_s)

    def gamma(self):
        """
        Calculate the Gamma of the option.
        Measures the sensitivity of the option price to changes in delta.
        dprice/ddelta = d^2 price / d^2 S
        USES FINITE-ELEMENT DERIVATIVES
        :return: Gamma value
        """
        original_S0 = self.S0
        price_up = self.price(S0=original_S0 + self.epsilon_s)
        price_down = self.price(S0=original_S0 - self.epsilon_s)
        price_current = self.price(S0=original_S0)
        self.S0 = original_S0
        return (price_up - 2 * price_current + price_down) / (self.epsilon_s ** 2)

    def vega(self):
        """
        Calculate the Vega of the option.
        Measures the sensitivity of the option price to changes in volatility.
        d price / d sigma
        USES FINITE-ELEMENT DERIVATIVES
        :return: Vega value
        """
        original_sigma = self.sigma
        price_up = self.price(sigma=original_sigma + self.epsilon_sigma)
        price_down = self.price(sigma=original_sigma- self.epsilon_sigma)
        self.sigma = original_sigma
        return (price_up - price_down) / (2 * self.epsilon_sigma)

    def theta(self):
        """
        Calculate the Theta of the option.
        Measures the sensitivity of the option price to the passage of time (time decay).
        d price/ dt
        USES FINITE-ELEMENT DERIVATIVES
        :return: Theta value
        """

        original_T = self.T
        price_up = self.price(T=original_T - self.epsilon_t)
        price_down = self.price(T=original_T + self.epsilon_t )
        self.T = original_T
        return (price_up - price_down) / (2*self.epsilon_t)
    
    def rho(self):
        """
        Calculate the Rho of the option.
        Measures the sensitivity of the option price to changes in the risk-free interest rate.
        dprice/dr
         USES FINITE-ELEMENT DERIVATIVES
        :return: Rho value
        """
        original_r = self.r
        price_up = self.price(r=original_r + self.epsilon_r)
        price_down = self.price(r=original_r - self.epsilon_r)
        self.r = original_r
        return (price_up - price_down) / (2 * self.epsilon_r)
 



class MCOpt_varying_volatility(OptionPricing):

    def __init__(self, S, K, T, r, sigma_function, option_type='call', samplesize = 1, epsilon=1e-1):
        """
        Initialize the Black-Scholes model parameters.
        
        :param S: Current stock price
        :param K: Option strike price
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate (annual)
        :param sigma: Volatility of the underlying asset (annual)
        :param option_type: 'call' for a call option, 'put' for a put option
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma_function = sigma_function
        self.option_type = option_type
        self.numsteps = int(self.T*365) # 1 step a day
        self.samplesize = samplesize
        self.epsilon = epsilon
        self.epsilon_s = epsilon*self.S
        self.epsilon_t = epsilon*self.T
        self.epsilon_r = epsilon*np.abs(r)


    def simulate_GBM(self):
    
        dt = self.T /self.numsteps
        t = np.linspace(0, self.T, self.numsteps)
        S = self.S
        for i in range(self.numsteps):
            W = np.random.standard_normal()*np.sqrt(dt)
            sigma = self.sigma_function(S,t[i])
            S*= np.exp((self.r - 0.5 * sigma**2)*dt + sigma*W)
        return float(S)
    
    def price(self, S=None, T=None, r=None, return_price_std = False):
        if S is None: S = self.S
        if T is None:T = self.T
        if r is None:r = self.r
        
        S_samples = np.zeros(self.samplesize)
        for i in range(self.samplesize):
            self.S = S
            self.T = T
            self.r = r
            S_samples[i] = self.simulate_GBM()

        if self.option_type == 'call':
            option_prices = np.maximum(S_samples - self.K, 0)
        elif self.option_type == 'put':
            option_prices = np.maximum(self.K - S_samples, 0)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        option_prices *= np.exp(-self.r * self.T)
        average_price = np.mean(option_prices)
        std_price = np.std(option_prices)

        if return_price_std: return average_price , std_price
        else: return average_price

    # def price(self, return_price_std = False):
    #     # S_sample = np.zeros(self.samplesize)
    #     # for i in range(self.samplesize):
    #     #     S_sample[i] = self.simulate_GBM()
        
    #     # #now, we can make an array of option prices
    #     # if self.option_type == 'Call':
    #     #     prices = np.maximum(S_sample - self.K, 0)
    #     # elif self.option_type == 'put':
    #     #     option_prices = np.maximum(self.K - S_sample, 0)
    #     # else:
    #     #     raise ValueError("Option type must be either 'call' or 'put'.")

    #     # # Discount to present value
    #     # option_prices *= np.exp(-self.r * self.T)

    #     # # Calculate the average and standard deviation of the option prices
    #     # average_price = np.mean(option_prices)
    #     # price_std_deviation = np.std(option_prices)

    #     # if return_price_std: return average_price, price_std_deviation
    #     # else: return average_price
    def delta(self):
        """
        Calculate the Delta of the option.
        Measures the sensitivity of the option price to changes in the underlying asset price.
        dprice/dS
        USES FINITE-ELEMENT DERIVATIVES
        :return: Delta value
        """
        original_S = self.S
        price_up = self.price(S=original_S + self.epsilon_s)
        price_down = self.price(S=original_S- self.epsilon_s)
        #S returns to original value
        self.S = original_S
        return (price_up - price_down) / (2 * self.epsilon_s)
    

        
    
    def gamma(self):
        """
        Calculate the Gamma of the option.
        Measures the sensitivity of the option price to changes in delta.
        dprice/ddelta = d^2 price / d^2 S
        USES FINITE-ELEMENT DERIVATIVES
        :return: Gamma value
        """
        original_S = self.S
        price_up = self.price(S=original_S + self.epsilon_s)
        price_down = self.price(S=original_S - self.epsilon_s)
        price_current = self.price(S=original_S)
        self.S = original_S
        return (price_up - 2 * price_current + price_down) / (self.epsilon_s ** 2)
    
    def vega(self):
        """
        Calculate the Vega of the option.
        Measures the sensitivity of the option price to changes in volatility.
        d price / d sigma
        USES FINITE-ELEMENT DERIVATIVES
        :return: Vega value
        """
        print('since we have a local volatility surface, this method can NOT calculate vega' )
  
    def theta(self):
        """
        Calculate the Theta of the option.
        Measures the sensitivity of the option price to the passage of time (time decay).
        d price/ dt
        USES FINITE-ELEMENT DERIVATIVES
        :return: Theta value
        """
 
        original_T = self.T
        price_up = self.price(T=original_T - self.epsilon_t)
        price_down = self.price(T=original_T + self.epsilon_t )
        self.T = original_T
        return (price_up - price_down) / (2*self.epsilon_t)
    
    def rho(self):
        """
        Calculate the Rho of the option.
        Measures the sensitivity of the option price to changes in the risk-free interest rate.
        dprice/dr
         USES FINITE-ELEMENT DERIVATIVES
        :return: Rho value
        """
        original_r = self.r
        price_up = self.price(r=original_r + self.epsilon_r)
        price_down = self.price(r=original_r - self.epsilon_r)
        self.r = original_r
        return (price_up - price_down) / (2*self.epsilon_r)

