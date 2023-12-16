import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import fsolve, minimize

PLOT_PNG_NAME = "sd-mu_plot.png"

def get_risk_param_and_money():
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
        return int_param * 0.02, float_money, True
      case 'n':
        return int_param * 0.02, float_money, False
    print("Please type either 'Y' or 'N'")
    

def find_r_and_C(f_location):
  '''
  compute and returns mean vector r and correlation matrix cov from csv data
  '''
  # read csv file and store in data
  data = pd.read_csv(f_location, parse_dates=['Date'], date_format='%d/%m/%Y')

  # label stock names to be read by for loop
  stock_names = ("BHP", "CSL", "NAB", "TCL", "TLS")

  # create matrix of arrays of all 5 stock returns
  returns = []
  for i in stock_names:
    returns.append((data[i]/data[i].shift(1) - 1)[1:])
  returns = np.array(returns)

  # calculate mean vector and covarience matrix
  return np.mean(returns, axis=1), np.cov(returns)

def compute_coefficients(r, cov):
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

def crit_line(t, r, cov, ss, alpha, beta, e):
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
  x_init = alpha # arbitrary starting point
  return minimize(obj_Z, x_init, bounds=bounds, constraints=con).x # scipy

def compute_mean_sd(x, r, cov):
  '''
  computes portfolio mean = x^T @ r, and sd = sqrt(x^T @ C @ x)
  assuming data is normally distributed, empirical return of this portfolio
  should have a 95% confidence interval = mean ± 2 * sd
  '''
  return np.dot(x, r), np.dot(x, cov @ x) ** 0.5

def generate_random_portfolios(r, cov):
  '''
  create 5000 random portfolios to plot onto graph
  returns tuple of means and tuple of sds of 5000 random portfolios
  '''

  # define arbitrary bounds
  lower = -100
  upper = 100
  number_of_generates = 5000
  # plot 5000 random portfolios

  # generate 5 random numbers between arbitrary bounds, 5000 times
  # after scaling them down, these will represent the weights of each asset
  i = 0
  mu_rand_list = []
  sig_rand_list = []
  while i < number_of_generates:
    x_rand = np.random.uniform(low=lower, high=upper, size=5)
    x_rand /= x_rand.sum() # divide each by sum so sum of 5 numbers = 1
    mu_rand = np.dot(x_rand, r) # mu = x^Tr
    sig_rand = np.dot(x_rand, cov @ x_rand) ** 0.5 # sigma = sqrt(x^TCx)
    if(sig_rand > 0.025): # if outside recomended domain, re-roll and do it again
      continue # otherwise random outlier portfolios ruin the scale of graph
    mu_rand_list.append(mu_rand)
    sig_rand_list.append(sig_rand)
    i += 1
  return tuple(sig_rand_list), tuple(mu_rand_list)


def plot_mu_sigma(a, b, d, mean, sd, r, cov):
  '''
  create and save png of plot to be displayed in html document with:
  5000 random portfolios
  Mean varience frontier (MVF)
  Efficiency Frontier (EF)
  User's portfolio, calculated previously
  Minimum risk portfolio (vertex of hyperbola)
  '''
  sig_rand_tup, mu_rand_tup = generate_random_portfolios(r, cov)
  plt.plot(sig_rand_tup, mu_rand_tup, 'r.', label="5000 rands")

  # MVF and EF respectivly
  # MVF: sigma^2 = 1/a + a/d(mu -b/a)^2
  mu = np.linspace(-0.001, 0.002)
  sigma = (1 / a + a * (mu - b/a)**2 / d) ** 0.5
  plt.plot(sigma, mu, 'k--', label="MVF")

  # EF: top half of hyperbola --> mu >= b/a
  mu = np.linspace(b/a, 0.002)
  sigma = (1 / a + a * (mu - b/a)**2 / d) ** 0.5
  plt.plot(sigma, mu, 'm', label="EF")

  # user's portfolio
  plt.plot(sd, mean, 'kx', label="Q3 portfolio")

  # minimum risk portfolio
  plt.plot(a**-0.5, b/a, 'gx', label="Min Risk")

  plt.xlabel('σ - Risk')
  plt.ylabel('μ - Expected Return')
  plt.legend()
  plt.tight_layout()
  plt.savefig(PLOT_PNG_NAME)


def main():
  t, money, ss = get_risk_param_and_money()
  r, cov = find_r_and_C('project_data.csv')
  a, b, c, d, e, alpha, beta, c_inv = compute_coefficients(r, cov)
  x = crit_line(t, r, cov, ss, alpha, beta, e)
  mean, sd = compute_mean_sd(x, r, cov)
  plot_mu_sigma(a, b, d, mean, sd, r, cov)

if __name__ == "__main__":
  main()
