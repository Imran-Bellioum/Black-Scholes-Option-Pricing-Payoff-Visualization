import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Compute the Black–Scholes price for European call or put options.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )
    elif option_type == "put":
        price = (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * norm.cdf(-d1)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def option_payoff(S, K, option_type="call"):
    """
    Compute the payoff of a European option at expiry.
    """
    if option_type == "call":
        return np.maximum(S - K, 0)
    elif option_type == "put":
        return np.maximum(K - S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# --------------------------------------------------
# Parameters
# --------------------------------------------------

S = np.linspace(50, 150, 100)   # Underlying stock prices
K = 100                         # Strike price
T = 1.0                         # Time to maturity (years)
r = 0.05                        # Risk-free rate
sigma = 0.20                    # Volatility
option_type = "call"            # 'call' or 'put'

# --------------------------------------------------
# Compute option prices and payoffs
# --------------------------------------------------

prices = [
    black_scholes(s, K, T, r, sigma, option_type)
    for s in S
]

payoffs = option_payoff(S, K, option_type)

# --------------------------------------------------
# Plot results
# --------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(
    S,
    prices,
    label="Option Price (Black–Scholes)",
    color="blue"
)
plt.plot(
    S,
    payoffs,
    label="Payoff at Expiry",
    color="green",
    linestyle="--"
)

plt.title(f"{option_type.capitalize()} Option Price vs Payoff at Expiry")
plt.xlabel("Underlying Stock Price ($)")
plt.ylabel("Value ($)")
plt.legend()
plt.grid(True)
plt.show()
