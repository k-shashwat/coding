#!/usr/bin/env python3
"""
main.py

End-to-end implementation for "Construction of an Infrastructure Financing Portfolio"
- Synthetic data generation (default) or load real price CSV (Date index + tickers)
- Preprocessing (log returns, winsorization)
- Estimate expected returns and (shrunk) covariance
- Compute efficient frontier, min-variance and max-Sharpe portfolios
- Simple stress testing using factor exposures (if provided)
- Save outputs: frontier plot, weight CSVs, summary CSV

Usage:
    python main.py            # runs using synthetic data
    python main.py --data-path prices.csv
    python main.py --data-path prices.csv --output-dir my_outputs --rf 0.01

Author: Shashwat (adapted)
"""
import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------
# Utility & Data generation
# ---------------------------
def generate_synthetic_prices(n_assets=100, years=20, freq_per_year=12, seed=42):
    np.random.seed(seed)
    n_periods = years * freq_per_year
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_periods, freq='M')

    # simple factor model for returns
    mu_f = np.array([0.004, 0.0, 0.0])  # monthly factor drifts
    Sigma_f = np.diag([0.0004, 0.0001, 0.0001])
    factors = np.random.multivariate_normal(mu_f, Sigma_f, size=n_periods)

    B = np.random.normal(0.0, 0.6, size=(n_assets, 3))
    eps_sigma = np.random.uniform(0.01, 0.06, size=n_assets)

    returns = np.zeros((n_periods, n_assets))
    for t in range(n_periods):
        eps = np.random.normal(0, eps_sigma)
        returns[t, :] = 0.002 + factors[t].dot(B.T) + eps

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    cols = [f"Asset_{i+1}" for i in range(n_assets)]
    df_prices = pd.DataFrame(prices, index=dates, columns=cols)
    df_factors = pd.DataFrame(factors, index=dates, columns=["Market", "InterestShock", "InflationShock"])
    return df_prices, df_factors, B

def load_prices_csv(path):
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.sort_index()
    return df

# ---------------------------
# Preprocessing & Estimates
# ---------------------------
def compute_log_returns(df_prices):
    return np.log(df_prices).diff().dropna()

def winsorize_series(df_ret, lower_q=0.01, upper_q=0.99):
    lower = df_ret.quantile(lower_q)
    upper = df_ret.quantile(upper_q)
    return df_ret.clip(lower=lower, upper=upper, axis=1)

def annualize_mean_cov(mean_monthly, cov_monthly, freq=12):
    return mean_monthly * freq, cov_monthly * freq

def shrinkage_covariance(sample_cov, shrink=0.1):
    diag = np.diag(np.diag(sample_cov))
    return shrink * diag + (1 - shrink) * sample_cov

# ---------------------------
# Portfolio Optimization
# ---------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def minimize_variance(target_return, mean_returns, cov_matrix):
    n = len(mean_returns)
    x0 = np.repeat(1.0/n, n)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    def fun(x):
        return float(np.dot(x.T, np.dot(cov_matrix, x)))
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-10, 'maxiter':1000})
    return res.x if res.success else None

def efficient_frontier(mean_returns, cov_matrix, returns_grid):
    weights = []
    rets = []
    vols = []
    for r in returns_grid:
        w = minimize_variance(r, mean_returns, cov_matrix)
        if w is None:
            continue
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        weights.append(w)
        rets.append(ret)
        vols.append(vol)
    return np.array(weights), np.array(rets), np.array(vols)

def min_variance_portfolio(mean_returns, cov_matrix):
    n = len(mean_returns)
    x0 = np.repeat(1.0/n, n)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    def fun(x):
        return float(np.dot(x.T, np.dot(cov_matrix, x)))
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-10, 'maxiter':1000})
    return res.x if res.success else None

# ---------------------------
# Stress Testing
# ---------------------------
def stress_test(weights, mean_returns, cov_matrix, exposures=None, factors_df=None, shock=None):
    # shock is dict of factor name -> monthly delta (e.g., {"InterestShock": -0.005})
    if exposures is not None and factors_df is not None and shock is not None:
        factor_means = factors_df.mean().values
        factor_names = list(factors_df.columns)
        factor_means_shocked = factor_means.copy()
        for i, name in enumerate(factor_names):
            if name in shock:
                factor_means_shocked[i] += shock[name]
        delta = exposures.dot((factor_means_shocked - factor_means))
        stressed_mean = mean_returns + delta
    else:
        # conservative -1% monthly shock across assets if no factor info
        stressed_mean = mean_returns - 0.01
    return portfolio_performance(weights, stressed_mean, cov_matrix)

# ---------------------------
# I/O & Plotting
# ---------------------------
def save_weights(weights, tickers, path):
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df.to_csv(path, index=False)

def plot_efficient_frontier(vols, rets, vol_min, ret_min, vol_shp, ret_shp, outpath):
    plt.figure(figsize=(8,5))
    plt.plot(vols, rets, label="Efficient frontier")
    plt.scatter(vol_min, ret_min, marker='o', label="Min variance")
    plt.scatter(vol_shp, ret_shp, marker='*', label="Max Sharpe")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_path:
        print(f"Loading data from {args.data_path}")
        df_prices = load_prices_csv(args.data_path)
        df_factors = None
        exposures = None
    else:
        print("Generating synthetic price data")
        df_prices, df_factors, exposures = generate_synthetic_prices(
            n_assets=args.n_assets, years=args.years, freq_per_year=args.freq_per_year, seed=args.seed
        )

    # Preprocess
    df_ret = compute_log_returns(df_prices)
    df_ret = winsorize_series(df_ret, lower_q=args.winsor_lower, upper_q=args.winsor_upper)

    mean_monthly = df_ret.mean()
    cov_monthly = df_ret.cov()
    mean_annual, cov_annual = annualize_mean_cov(mean_monthly, cov_monthly, freq=args.freq_per_year)
    cov_annual_shrunk = shrinkage_covariance(cov_annual, shrink=args.shrinkage)

    # Efficient frontier
    returns_grid = np.linspace(mean_annual.min(), mean_annual.max(), 50)
    weights_eff, rets_eff, vols_eff = efficient_frontier(mean_annual.values, cov_annual_shrunk.values, returns_grid)

    # Max Sharpe (approx on frontier)
    rf = args.rf
    sharpe_scores = (rets_eff - rf) / vols_eff
    idx = np.nanargmax(sharpe_scores)
    w_max_sharpe = weights_eff[idx]
    ret_shp, vol_shp = portfolio_performance(w_max_sharpe, mean_annual.values, cov_annual_shrunk.values)
    sharpe_max = (ret_shp - rf) / vol_shp

    # Min variance
    w_min = min_variance_portfolio(mean_annual.values, cov_annual_shrunk.values)
    ret_min, vol_min = portfolio_performance(w_min, mean_annual.values, cov_annual_shrunk.values)

    # Save weights & summary
    tickers = df_prices.columns.tolist()
    save_weights(w_max_sharpe, tickers, os.path.join(args.output_dir, "weights_max_sharpe.csv"))
    save_weights(w_min, tickers, os.path.join(args.output_dir, "weights_min_variance.csv"))

    plot_efficient_frontier(vols_eff, rets_eff, vol_min, ret_min, vol_shp, ret_shp, os.path.join(args.output_dir, "efficient_frontier.png"))

    # Stress test
    if df_factors is not None and exposures is not None:
        shock = {"InterestShock": args.stress_interest, "InflationShock": args.stress_inflation}
        stressed_shp = stress_test(w_max_sharpe, mean_annual.values, cov_annual_shrunk.values, exposures=exposures, factors_df=df_factors, shock=shock)
        stressed_min = stress_test(w_min, mean_annual.values, cov_annual_shrunk.values, exposures=exposures, factors_df=df_factors, shock=shock)
    else:
        shocked = (-0.01,)
        stressed_shp = stress_test(w_max_sharpe, mean_annual.values, cov_annual_shrunk.values)
        stressed_min = stress_test(w_min, mean_annual.values, cov_annual_shrunk.values)

    summary = {
        "n_assets": len(tickers),
        "years": args.years,
        "rf": rf,
        "ret_max_sharpe": float(ret_shp),
        "vol_max_sharpe": float(vol_shp),
        "sharpe_max": float(sharpe_max),
        "ret_min_variance": float(ret_min),
        "vol_min_variance": float(vol_min),
        "stressed_max_sharpe_return": float(stressed_shp[0]),
        "stressed_min_var_return": float(stressed_min[0])
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "summary_report.csv"), index=False)

    print("Saved outputs to", args.output_dir)
    print(f"Max Sharpe annual return: {ret_shp:.2%}, vol: {vol_shp:.2%}, Sharpe: {sharpe_max:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Markowitz infrastructure portfolio optimization")
    p.add_argument("--data-path", type=str, default=None, help="CSV file of prices (Date index + tickers)")
    p.add_argument("--output-dir", type=str, default="outputs", help="Directory to save outputs")
    p.add_argument("--rf", type=float, default=0.01, help="Annual risk free rate (decimal)")
    p.add_argument("--n-assets", type=int, default=100)
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--freq-per-year", type=int, default=12)
    p.add_argument("--shrinkage", type=float, default=0.1)
    p.add_argument("--winsor-lower", type=float, default=0.01)
    p.add_argument("--winsor-upper", type=float, default=0.99)
    p.add_argument("--stress-interest", type=float, default=-0.005, help="Monthly interest factor shock for stress test")
    p.add_argument("--stress-inflation", type=float, default=-0.003, help="Monthly inflation factor shock for stress test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
