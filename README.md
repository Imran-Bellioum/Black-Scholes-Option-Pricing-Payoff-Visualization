# Black-Scholes Option Pricing Model

This project implements the Black-Scholes model for pricing European call and put options.

The program calculates theoretical option prices across a range of underlying stock prices and compares them with the option payoff at expiry. It then visualises the difference between the Black-Scholes option value before expiry and the final payoff at maturity.

## Project Overview

The Black-Scholes model is a mathematical model used to estimate the fair value of European options.

A European option can only be exercised at expiry. The model uses inputs such as the current stock price, strike price, time to maturity, risk-free interest rate, and volatility to calculate the theoretical value of an option.

This project focuses on:

* Implementing the Black-Scholes formula in Python
* Pricing European call and put options
* Calculating option payoffs at expiry
* Visualising option price versus payoff

## Black-Scholes Formula

For a European call option, the Black-Scholes price is:

$$
C = S N(d_1) - K e^{-rT} N(d_2)
$$

For a European put option, the price is:

$$
P = K e^{-rT} N(-d_2) - S N(-d_1)
$$

where:

$$
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}
$$

and:

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

## Variable Definitions

* $S$ is the current underlying stock price
* $K$ is the strike price
* $T$ is the time to maturity in years
* $r$ is the risk-free interest rate
* $\sigma$ is the volatility of the underlying asset
* $N(x)$ is the cumulative distribution function of the standard normal distribution
* $C$ is the European call option price
* $P$ is the European put option price

## Payoff at Expiry

For a call option, the payoff at expiry is:

$$
\max(S - K, 0)
$$

For a put option, the payoff at expiry is:

$$
\max(K - S, 0)
$$

The payoff shows the value of the option at maturity, while the Black-Scholes price represents the theoretical value before expiry.

## Features

* Prices European call and put options
* Uses the Black-Scholes closed-form formula
* Calculates option payoffs at expiry
* Plots option price against payoff
* Allows the user to change key option parameters:

  * Stock price range
  * Strike price
  * Time to maturity
  * Risk-free rate
  * Volatility
  * Option type

## Technologies Used

* Python
* NumPy
* Matplotlib
* SciPy

## Example Parameters

The current version uses the following example inputs:

* Underlying stock price range: 50 to 150
* Strike price: 100
* Time to maturity: 1 year
* Risk-free rate: 5%
* Volatility: 20%
* Option type: Call

These parameters can be changed directly in the script.

## Example Output

The program produces a graph comparing:

* The theoretical Black-Scholes option price
* The payoff at expiry

For a call option, the payoff is zero when the stock price is below the strike price and increases linearly once the stock price rises above the strike price.

The Black-Scholes price is usually above the payoff before expiry because it includes time value. This reflects the possibility that the option may become more valuable before maturity.

## How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/black-scholes-option-pricing.git
```

Install the required packages:

```bash
pip install numpy matplotlib scipy
```

Run the script:

```bash
python black_scholes.py
```

## Financial Interpretation

The graph helps show the difference between an option's intrinsic value and its theoretical price before expiry.

The payoff curve shows what the option would be worth at maturity. The Black-Scholes curve includes both intrinsic value and time value, which is why the option price can be positive even when the option is currently out of the money.

This is especially visible for call options when the underlying stock price is below the strike price. Even though the immediate payoff would be zero, the option still has value because there is a chance that the stock price may rise before expiry.

## Limitations

This is a simplified implementation of the Black-Scholes model and should not be treated as a real trading or investment tool.

Important limitations include:

* Assumes constant volatility
* Assumes a constant risk-free interest rate
* Assumes no dividends
* Assumes markets are frictionless
* Assumes returns are lognormally distributed
* Only applies directly to European options
* Does not account for transaction costs, liquidity, or market impact

## Possible Improvements

Future extensions could include:

* Adding support for dividend-paying stocks
* Calculating option Greeks such as Delta, Gamma, Vega, Theta, and Rho
* Comparing Black-Scholes prices with Monte Carlo option pricing
* Adding implied volatility calculations
* Building an interactive dashboard using Streamlit
* Allowing user input from the command line
* Plotting both call and put prices on the same graph

## Purpose

This project was created as a beginner quantitative finance project to practise option pricing, mathematical finance, Python programming, and data visualisation.
