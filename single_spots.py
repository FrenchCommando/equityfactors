import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from scipy.optimize import minimize

import yfinance as yf

import matplotlib.ticker as mtick
import matplotlib
matplotlib.use('WebAgg')
# matplotlib.use('qtagg')


data_folder = "data"


def load_and_cache(ticker):
    filename = os.path.join(data_folder, f"{ticker}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        return df
    start = dt.date(2025, 1, 1)
    end = dt.date.today()
    ticker = yf.Ticker(ticker)

    historical_data = ticker.history(start=start, end=end)
    historical_data.to_csv(filename)
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df


def get_returns(ticker):
    if isinstance(ticker, list):
        out = pd.DataFrame.from_dict({one_ticker: get_returns(ticker=one_ticker) for one_ticker in ticker})
        return out.dropna(axis=1)
    # print(ticker)
    df = load_and_cache(ticker=ticker)
    df_close = df['Close']
    df_r = df_close.pct_change().dropna()
    return df_r

def get_components():
    file_path = "holdings-daily-us-en-spy.xlsx"
    df = pd.read_excel(file_path, skiprows=4).head(2)
    return list(map(lambda x: x.replace(".", "-"), df['Ticker'].dropna().values)), list(map(lambda x: x * 0.01, df['Weight'].dropna().values))

def get_all_components():
    names, weights = get_components()
    df_all = get_returns(ticker=names)
    return df_all, weights


def get_decomposition(data, weights):
    portfolio = np.dot(data, weights)
    n = len(weights)
    # print(portfolio)
    portfolio_var = np.var(portfolio)
    portfolio_std = np.sqrt(portfolio_var)
    print("Var", portfolio_var)
    print("Std", portfolio_std)
    fig, ax = plt.subplots(num="Portfolio")
    # cum_return = (portfolio + 1).cumprod() - 1
    # print(cum_return)
    # plt.plot(list(enumerate(cum_return)), cum_return, label="PnL")
    plt.plot(data.index, list(portfolio), label="PnL")
    plt.axhline(y=portfolio_std, label="Std")

    covar = np.cov(data.T)
    # print(covar)
    print(covar.shape)

    eigenvalues, eigenvectors = np.linalg.eig(covar)
    print(eigenvalues.shape)
    print(eigenvectors.shape)
    print(eigenvalues)
    print(eigenvectors.T[0])

    for i, v in enumerate(eigenvectors.T):
        eigen_value = eigenvalues[i]
        if eigen_value < 1e-4:
            continue
        n_i = np.dot(v, v)
        p_i = np.dot(data, v)
        var_i = np.var(p_i)
        std_i = np.sqrt(var_i)
        # print(f"N{i}  ", n_i)
        # print(f"Var{i}  ", var_i)
        # print(f"Var{i}*n", var_i * (n))
        print(f"Std  {i}", std_i)
        print(f"EigQr{i}", np.sqrt(eigen_value))
        # print(f"Eigen{i}", eigen_value)
        print()
        plt.plot(data.index, list(p_i), label=f"PnL{i}")
        plt.axhline(y=std_i, label=f"Std{i}")


    # result from optimization
    def variance(w):
        n_w = np.dot(w, w)
        p_w = np.dot(data, w)
        var_w = np.var(p_w) / n_w
        return - var_w  #+ 1e4 * np.abs(n_w - 1)

    x0 = np.zeros_like(weights)
    w_opt = minimize(variance, x0, method='nelder-mead')
    print(w_opt)
    print("Optimized", w_opt.x)
    print("OptimizedN", np.dot(w_opt.x, w_opt.x))
    print("OptimizedVar", variance(w=w_opt.x) / np.dot(w_opt.x, w_opt.x))
    print("OptimizedStd", np.sqrt(np.var(np.dot(data, w_opt.x)) / np.dot(w_opt.x, w_opt.x)))

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.legend()
    plt.minorticks_on()
    plt.grid(visible=True, alpha=0.8, which='major')
    plt.grid(visible=True, alpha=0.2, which='minor')


def main():
    # ticker_symbols = ["AAPL", "AMZN", "NVDA", "CAT"]
    # df = get_returns(ticker=ticker_symbols)
    # print(df)
    df_all, weights = get_all_components()
    get_decomposition(data=df_all, weights=weights)
    # print(df_all.head().T)
    # df_all.to_csv("all.csv")
    plt.show()


if __name__ == '__main__':
    main()
