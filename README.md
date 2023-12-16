
# Portfolio Generator

This project uses [Markowitz's Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory) (MPT) to generate an optimal portfolio given a set of stocks. I used it to better my knowledge in functional programing with python and to explore the scipy, numpy, pandas and matplotlib librarys. I was motivated to write this as an assignment for my portfolio optimisation and financial mathematics unit [MATH2070](https://www.sydney.edu.au/units/MATH2070). Among all of MPT's assumptions, this program assumes that stock data is normally distributed.

I am improving this project by working on:
1. Adding a risk free asset stock with user defined lending/borrowing rates,
2. A method to compute the market portfolio,
3. Outputting each stock's beta,
4. Plotting the security market line.

## Features

- Takes user's risk tolerance level, budget, and short selling preference as input,
- Generates a portfolio that maximizes expected return for a given level of risk,
- Outputs a html report with portfolio specifications, statistics, the optimal portfolio, and a graphical interpretation of the portfolio.

## Requirements

- Python 3.7 or later,
- NumPy,
- SciPy.

## Usage

1. Download some stock data with the following columns structure: Date, Stock 1, Stock 2, ... (in this order; no other columns) and rows as stock prices for each day; or just download `stock_prices.csv` above,
2. Download `portfolio_generator.py` in the same directory as the formatted stock data,
3. Navigate to the project directory and type: `python3 portfolio_generator.py` into the terminal.

If you do not wish to download this software, the sample stocks `stock_prices.csv` were used to create a sample report which you can read from [`report.html``](https://ericdotmills5.github.io/portfolio-generator/).

## Disclaimer
This project is for educational purposes only. It should not be used for real-world investment decisions. The author of this project is not responsible for any financial losses.

