import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import math

import matplotlib.pyplot as plt

import plotly.graph_objects as plotly

def option_chain(ticker):
    asset = yf.Ticker(ticker)
    expirations = asset.options
    chains = pd.DataFrame()

    for exp in expirations:
        opt = asset.option_chain(exp)

        calls = opt.calls
        puts = opt.puts

        calls['optionType'] = 'call'
        puts['optionType'] = 'put'

        chain = pd.concat([calls,puts])
        chain['expiration'] = pd.to_datetime(exp) + pd.DateOffset(hours=23, minutes=59, seconds=59)

        chains = pd.concat([chains,chain])

    chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
    return chains


def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes call option price."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call_price


def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate vega (sensitivity to volatility) for Black-Scholes."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * norm_pdf(d1)
    return vega


def implied_volatility_newton_raphson(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    max_iterations: int = 10000,
) -> float | None:
    """Calculate implied volatility using Newton-Raphson method."""
    # Initial guess
    sigma = 0.3
    tolerance = 1e-6

    for _ in range(max_iterations):
        # Calculate option price and vega at current sigma
        calculated_price = black_scholes_call(S, K, T, r, sigma)
        vega = black_scholes_vega(S, K, T, r, sigma)

        price_diff = calculated_price - market_price

        if abs(price_diff) < tolerance:
            return sigma

        if abs(vega) < 1e-10:  # Avoid division by zero
            return None

        sigma = sigma - price_diff / vega

        # Keep sigma positive and reasonable
        sigma = max(0.001, min(sigma, 5.0))

    return None


def compute_custom_iv(df, S, r=0.05):
    """Attach custom implied vols to a DataFrame of option quotes."""
    ivs = []
    today = dt.datetime.today()

    for _, row in df.iterrows():
        K = row["strike"]
        T = (row["expiration"] - today).days / 365.0
        market_price = row["lastPrice"]

        if T <= 0 or market_price <= 0:
            ivs.append(None)
            continue

        sigma = implied_volatility_newton_raphson(S, K, T, r, market_price)
        ivs.append(sigma)

    df = df.copy()
    df["newton_iv"] = ivs
    return df

options = option_chain("SPY")

calls = options[options['optionType'] == 'call']

set(calls.expiration)

S = yf.Ticker("SPY").history(period="1d")["Close"].iloc[-1]
calls_at_expiry = calls[calls["expiration"] == pd.Timestamp("2025-10-10 23:59:59")]

calls_at_expiry_iv = compute_custom_iv(calls_at_expiry, S)

filtered_calls_at_expiry = calls_at_expiry_iv[calls_at_expiry_iv.newton_iv >= 0.001]

"""
filtered_calls_at_expiry[["strike", "newton_iv"]].dropna().set_index("strike").plot(
    title="Custom Implied Volatility Skew", figsize=(7,4)
)
"""

calls_at_strike = options[options["strike"] == 700.0]

calls_at_strike_iv = compute_custom_iv(calls_at_strike, S)

filtered_calls_at_strike = calls_at_strike_iv[calls_at_strike_iv.newton_iv >= 0.001]

calls = compute_custom_iv(calls, S)

surface = (
    calls[['daysToExpiration', 'strike', 'newton_iv']]
    .pivot_table(values='newton_iv', index='strike', columns='daysToExpiration')
    .dropna()
)

# create the figure object
fig = plt.figure(figsize=(10, 8))

# add the subplot with projection argument
ax = fig.add_subplot(111, projection='3d')

# get the 1d values from the pivoted dataframe
x, y, z = surface.columns.values, surface.index.values, surface.values

# return coordinate matrices from coordinate vectors
X, Y = np.meshgrid(x, y)

# set labels
ax.set_xlabel('Days to expiration')
ax.set_ylabel('Strike price')
ax.set_zlabel('Implied volatility')
ax.set_title('Call implied volatility surface')

# plot
ax.plot_surface(X, Y, z)

plt.show()
