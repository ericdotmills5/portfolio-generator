import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple

np.seterr(invalid='ignore')

PLOT_PNG_NAME = "sd-mu_plot.png"
REPORT_NAME = "index.html"
T_SCALE = 0.035
STOCKS_CSV_NAME = "stock_prices.csv"
SIGMA_GRAPH_RANGE = 0.025
NUMBER_OF_RANDOM_PORTFOLIOS = 5000


def get_risk_param_and_money() -> Tuple[float, float, bool]:
  '''
  get risk parameter t from user and their wealth 
  and their short selling prefernce
  '''
  # get risk parameter
  while True:
    param = input("Enter risk tolerance level [1-10]: ").strip()

    try:
      int_param = int(param)
    except ValueError:
      print("Please enter an integer between 1-10 inclusive")
      continue
    
    if (int_param < 1 or int_param > 10):
      print("Please enter an integer between 1-10 inclusive")
      continue

    break

  # get wealth
  while True:
    money = input("Enter budget (eg 521.25): ").strip()

    try:
      float_money = float(money)
    except ValueError:
      print("Please enter an a positive decimal number")
      continue

    if(float_money <= 0):
      print("Please enter an a positive decimal number")
      continue
    
    break
  
  while True:
    ss = input("Allow Short Selling (Y/N): ").strip().lower()
    match ss:
      case 'y':
        return int_param * T_SCALE, float_money, True
      case 'n':
        return int_param * T_SCALE, float_money, False
    print("Please type either 'Y' or 'N'")
    

def find_r_and_C(f_location: str) -> Tuple[np.ndarray, np.ndarray, Tuple[str]]:
  '''
  compute and returns mean vector r and correlation matrix cov from csv data
  '''
  # read csv file and store in data
  data = pd.read_csv(f_location, parse_dates=['Date'], date_format='%d/%m/%Y')

  # label stock names to be read by for loop
  stock_names = tuple(data.columns.tolist()[1:])

  if len(stock_names) <= 1:
    raise Exception("CSV File Provided has < 2 stocks to compare")

  # create matrix of arrays of all stock returns
  returns = []
  for i in stock_names:
    returns.append((data[i]/data[i].shift(1) - 1)[1:])
  returns = np.array(returns)

  # calculate mean vector and covarience matrix
  return np.mean(returns, axis=1), np.cov(returns), stock_names

def compute_coefficients(r: np.ndarray, cov: np.ndarray) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
  '''
  computes coefficients for critical line equation
  critical line determines optimal portfolio given a certain risk aversity
  returns all coefficients: a, b, c, d, e, alpha, beta, cov^-1
  '''
  # Cov^-1
  c_inv = np.linalg.inv(cov)

  # e = <1,1,1,...>
  e = np.ones(len(r))

  # compute a: Cov^-1 @ e -> a = e^T(C^-1 @ e)
  a = np.dot(e, c_inv @ e)

  # compute b: Cov^-1 @ e -> b = r^T(C^-1 @ e)
  b = np.dot(r, c_inv @ e)

  # compute c: c = r^T(cov^-1 @ r)
  c = np.dot(r, c_inv @ r)

  # compute d: d = ac - b^2
  d = a * c - b * b

  # compute alpha = Cov^-1 @ e / a
  alpha = c_inv @ e / a

  # compute beta b/aCov^-1 @ e = b * alpha --> beta = Cov^-1 @ r - b * alpha
  beta = c_inv @ r - b * alpha

  return a, b, c, d, e, alpha, beta, c_inv

def crit_line(t: float, r: np.ndarray, cov: np.ndarray, ss: bool, alpha: np.ndarray, beta: np.ndarray, e: np.ndarray):
  '''
  computes optimal proportions of budget x = alpha + t * beta
  '''
  if ss:
    return alpha + t * beta
  
  # define objective function: Z = -tx^Tr + 1/2x^TCx
  obj_Z = lambda x: -1 * t * np.dot(x, r) + np.dot(x, cov @ x) / 2

  # budget equality constraint x^te - 1 = 0
  budget = lambda x: np.dot(x, e) - 1

  # bounds: L = <0, 0, 0, 0, 0>, U = <inf, inf, inf, inf, inf>, ie x_i >= 0
  bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

  con = [{"type": "eq", "fun": budget}] # declaring budget is equality constraint
  x_init = alpha + t * beta # arbitrary starting point
  return minimize(obj_Z, x_init, bounds=bounds, constraints=con).x # scipy

def compute_mean_sd(x: np.ndarray, r: np.ndarray, cov: np.ndarray) -> Tuple[float, float]:
  '''
  computes portfolio mean = x^T @ r, and sd = sqrt(x^T @ C @ x)
  assuming data is normally distributed, empirical return of this portfolio
  should have a 95% confidence interval = mean ± 2 * sd
  '''
  return np.dot(x, r), np.dot(x, cov @ x) ** 0.5

def generate_random_portfolios(r: np.ndarray, cov: np.ndarray, ss: bool) -> Tuple[Tuple[float], Tuple[float]]:
  '''
  create 5000 random portfolios to plot onto graph
  returns tuple of means and tuple of sds of 5000 random portfolios
  '''

  # define arbitrary bounds
  lower = -100 if ss else 0
  upper = 100
  # plot 5000 random portfolios

  # generate 5 random numbers between arbitrary bounds, 5000 times
  # after scaling them down, these will represent the weights of each asset
  i = 0
  mu_rand_list = []
  sig_rand_list = []
  while i < NUMBER_OF_RANDOM_PORTFOLIOS:
    x_rand = np.random.uniform(low=lower, high=upper, size=len(r))
    x_rand /= x_rand.sum() # divide each by sum so sum of 5 numbers = 1
    mu_rand = np.dot(x_rand, r) # mu = x^Tr
    sig_rand = np.dot(x_rand, cov @ x_rand) ** 0.5 # sigma = sqrt(x^TCx)
    if(sig_rand > 0.025): # if outside recomended domain, re-roll and do it again
      continue # otherwise random outlier portfolios ruin the scale of graph
    mu_rand_list.append(mu_rand)
    sig_rand_list.append(sig_rand)
    i += 1
  return tuple(sig_rand_list), tuple(mu_rand_list)


def plot_mu_sigma(a: float, b: float, d: float, mean: float, sd: float, r: np.ndarray, cov: np.ndarray, ss: bool) -> None:
  '''
  create and save png of plot to be displayed in html document with:
  5000 random portfolios
  Mean varience frontier (MVF)
  Efficiency Frontier (EF)
  User's portfolio, calculated previously
  Minimum risk portfolio (vertex of hyperbola)
  '''
  sig_rand_tup, mu_rand_tup = generate_random_portfolios(r, cov, ss)
  plt.plot(sig_rand_tup, mu_rand_tup, 'r.', label=f"{NUMBER_OF_RANDOM_PORTFOLIOS} rands^")

  # MVF and EF respectivly
  # MVF: sigma^2 = 1/a + a/d(mu -b/a)^2
  sigma = np.linspace(0, SIGMA_GRAPH_RANGE)
  mu = b / a - (d * (sigma ** 2 - 1 / a) / a) ** 0.5

  #mu = np.linspace(-0.001, 0.002)
  #sigma = (1 / a + a * (mu - b/a)**2 / d) ** 0.5
  plt.plot(sigma, mu, 'k--', label="MVF")

  # EF: top half of hyperbola --> mu >= b/a
  #mu = np.linspace(b/a, 0.002)
  #sigma = (1 / a + a * (mu - b/a)**2 / d) ** 0.5

  mu = b / a + (d * (sigma ** 2 - 1 / a) / a) ** 0.5
  plt.plot(sigma, mu, 'm', label="EF")

  # user's portfolio
  plt.plot(sd, mean, 'kx', label="This portfolio")

  plt.xlabel('σ - Risk as a Propotion of Budget')
  plt.ylabel('μ - Expected Return as a Propotion of Budget')
  plt.legend()
  plt.tight_layout()
  plt.savefig(PLOT_PNG_NAME)

def create_abs_x(money: float, x: np.ndarray) -> Tuple[float]:
  '''
  scales up return propotion vector by budget
  '''
  abs_x = []
  for i in x:
    abs_x.append(i * money)
  return tuple(abs_x)

def create_returns_string(abs_x: Tuple[float], stock_names: Tuple[str]) -> str:
  '''
  turns the investment vector into a readable string for the html
  '''
  invest = ""
  for i, amount in enumerate(abs_x):
    invest += f"\t\t\t\t<p>{'Invest' if amount >= 0 else 'Short Sell'} ${abs(amount):.2f} into {stock_names[i]}</p>\n"
  return invest

def write_html(t: float, money: float, ss: bool, abs_mean: float, abs_sd: float, stock_string: Tuple[str]) -> None:
  '''
  writes html with computed information
  '''
  to_write = f"""<!DOCTYPE html>
<html>
<head>
  <title>Generated Portfolio Report</title>
  <style>
    .section {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 50px;
    }}
    .left {{
      width: 65%;
      text-align: left;
    }}
    .right {{
      width: 30%;
      text-align: right;
    }}
  </style>
</head>
<body>
  <div class="section">
    <div class="right">
      <h2>Portfolio Specifications</h2>
      <p>Risk indifference level: {round(t / T_SCALE)} out of 10 (t = {t:.3f})</p>
      <p>Budget: ${money:.2f}</p>
      <p>Short selling{" " if ss else " NOT "}ALLOWED</p>
      
      <h2>Portfolio statistics</h2>
      <p>This portfolio has an expected return of: ${abs_mean:.2f} per day</p>
      <p>This portfolio has risk of: ${abs_sd:.2f}</p>
      <p>Risk is defined as 1 standard deviation (68% CI)</p>

      <h2>Optimal portfolio</h2>
{stock_string}
      
    </div>
    
    <div class="left">
      <h2>Plot of randomly generated portfolios</h2>
    
      <img src="sd-mu_plot.png" alt="Description of the image">
      <p>^{NUMBER_OF_RANDOM_PORTFOLIOS} random portfolios generated{"" if ss else " with short selling NOT ALLOWED"}, and their risks/returns plotted.</p>
      <p>This portfolio was generated with <a href="https://en.wikipedia.org/wiki/Modern_portfolio_theory">Markowitz's Modern Portfolio Theory.</a></p>
      <p>This program is designed solely for educational purposes. The author explicitly disclaims any responsibility for any financial or other losses that may arise from the use of this program.</p>
    </div>

  
</body>
</html>
"""

  f = open(REPORT_NAME, "w")
  f.write(to_write)
  f.close()


def main() -> None:
  t, money, ss = get_risk_param_and_money()
  r, cov, stock_names = find_r_and_C(STOCKS_CSV_NAME)
  a, b, c, d, e, alpha, beta, c_inv = compute_coefficients(r, cov)
  x = crit_line(t, r, cov, ss, alpha, beta, e)
  mean, sd = compute_mean_sd(x, r, cov)
  abs_x = create_abs_x(money, x)
  plot_mu_sigma(a, b, d, mean, sd, r, cov, ss)
  stock_string = create_returns_string(abs_x, stock_names)
  write_html(t, money, ss, mean * money, sd * money, stock_string)
  print("Please open index.html to see the generated portfolio!")

if __name__ == "__main__":
  main()
