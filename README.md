
# Portfolio Generator

This project uses Markowitz's Modern Portfolio Theory to generate an optimal portfolio given a set of stocks. I used it to better my knowledge in functional programing with python as well as an introduction to portfolio optimisation. Among all of MPT's assumptions, this program assumes that stock data is normally distributed.

## Features

- Takes user's risk tolerance level, budget, and short selling preference as input
- Generates a portfolio that maximizes expected return for a given level of risk
- Outputs a report with portfolio specifications, statistics, and the optimal portfolio

## Requirements

- Python 3.7 or later
- NumPy
- SciPy

## Usage

1. Download some stock data with columns: Date, Stock 1, Stock 2, ... (in this order; no other columns) and rows as stock prices for each day; or just download "stock_prices.csv" above.
2. Download "portfolio_generator.py" in the same directory as the formatted stock data
3. Navigate to the project directory and run: "python3 portfolio_generator.py"

Disclaimer
This project is for educational purposes only. It should not be used for real-world investment decisions. The author of this project is not responsible for any financial losses.

