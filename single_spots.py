import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize

import yfinance as yf

# matplotlib.use('WebAgg')
matplotlib.use('qtagg')


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
    df = pd.read_excel(file_path, skiprows=4).head(50)
    df = df.loc[df["Name"] != "US DOLLAR"]
    # print(df)
    return list(map(lambda x: x.replace(".", "-"), df['Ticker'].dropna().values)), list(map(lambda x: x * 0.01, df['Weight'].dropna().values))

def get_all_components():
    names, weights = get_components()
    df_all = get_returns(ticker=names)
    return df_all, weights


def get_decomposition(data, weights):
    n = len(weights)
    norm_w = np.dot(weights, weights)
    norm_weights = weights / np.sqrt(norm_w)  # norm means euclidean norm == 1, not sum == 1
    norm_w_n = np.dot(norm_weights, norm_weights)
    portfolio = np.dot(data, norm_weights)
    # print(portfolio)
    portfolio_var = np.var(portfolio)
    portfolio_std = np.sqrt(portfolio_var)
    print("Norm", norm_w_n)
    print("Var", portfolio_var)
    print("Std", portfolio_std)
    fig, ax = plt.subplots(num="Portfolio")
    # cum_return = (portfolio + 1).cumprod() - 1
    # print(cum_return)
    # plt.plot(list(enumerate(cum_return)), cum_return, label="PnL")
    portfolio_color = 'black'
    # print(portfolio)
    plt.plot(data.index, np.cumsum(portfolio), label="PnL", color=portfolio_color)
    plt.axhline(y=portfolio_std, label="Std", linestyle='--', color=portfolio_color)

    date_cutoff = pd.Timestamp(dt.date(2025, 4, 1), tz="US/Eastern")
    # print(data)
    data0 = data.loc[:date_cutoff]
    data1 = data.loc[date_cutoff:]
    # print(data0)
    # print(data1)

    # use data0 for factors
    covar = np.cov(data0.T)
    # print(covar)
    print(covar.shape)

    eigenvalues, eigenvectors = np.linalg.eig(covar)
    print(eigenvalues.shape)
    print(eigenvectors.shape)
    print(eigenvalues)
    print(eigenvectors.T[0])

    portfolio0 = np.dot(data0, norm_weights)
    portfolio1 = np.dot(data1, norm_weights)
    portfolio_var0 = np.var(portfolio0)
    portfolio_var1 = np.var(portfolio1)
    portfolio_std0 = np.sqrt(portfolio_var0)
    portfolio_std1 = np.sqrt(portfolio_var1)
    out_data = dict(
        Portfolio=dict(
            Std=portfolio_std, Mean=np.mean(portfolio),
            Std0=portfolio_std0, Mean0=np.mean(portfolio0),
            Std1=portfolio_std1, Mean1=np.mean(portfolio1),
        ),
    )

    cmap = matplotlib.colormaps['Set1']  # .resampled(10)
    for i, v in enumerate(eigenvectors.T):
        eigen_value = eigenvalues[i]
        n_i = np.dot(v, v)
        p_i = np.dot(data, v)
        var_i = np.var(p_i)
        std_i = np.sqrt(var_i)

        p_i_0 = np.dot(data0, v)
        p_i_1 = np.dot(data1, v)
        var_i_0 = np.var(p_i_0)
        std_i_0 = np.sqrt(var_i_0)
        var_i_1 = np.var(p_i_1)
        std_i_1 = np.sqrt(var_i_1)

        out_data[f"{i}"] = dict(
            Std=std_i, Mean=np.mean(p_i),
            Std0=std_i_0, Mean0=np.mean(p_i_0),
            Std1=std_i_1, Mean1=np.mean(p_i_1),
        )

        # ignoring numerical noise, criteria on eigenvalue should be the same as criteria on std
        # if eigen_value < 1e-3:
        if std_i < 3.5e-2:  # look at the graph to adjust the cutoff
            continue

        print(f"N{i}  ", n_i)
        # print(f"Var{i}  ", var_i)
        # print(f"Var{i}*n", var_i * (n))
        print(f"Std  {i}", std_i)
        print(f"EigQr{i}", np.sqrt(eigen_value))

        # print(f"Eigen{i}", eigen_value)
        print()
        color = cmap(i)
        plt.plot(data.index, np.cumsum(p_i), label=f"PnL{i}", color=color)
        plt.axhline(y=std_i, label=f"Std{i}", linestyle='--', color=color)


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

    fig, ax = plt.subplots(num="Factors")
    for label, d_data in out_data.items():
        norm_data_std = d_data["Std"] * np.sqrt(252)
        if norm_data_std < 5e-1:
            continue
        sharpe = 252 * d_data["Mean"] / norm_data_std
        color = cmap(int(label)) if label != "Portfolio" else "black"
        plt.scatter(norm_data_std, sharpe, label=label, color=color)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.legend()
    plt.ylabel("Sharpe")
    plt.xlabel("Std(Annual)")
    plt.minorticks_on()
    plt.grid(visible=True, alpha=0.8, which='major')
    plt.grid(visible=True, alpha=0.2, which='minor')

    fig, ax = plt.subplots(num="FactorsCross")
    for label, d_data in out_data.items():
        norm_data_std_0 = d_data["Std0"] * np.sqrt(252)
        norm_data_std_1 = d_data["Std1"] * np.sqrt(252)
        if norm_data_std_0 < 3e-1:
            continue
        # rotation so that initial sharpe is positive
        sharpe_0 = 252 * d_data["Mean0"] / norm_data_std_0 * np.sign(d_data["Mean0"])
        sharpe_1 = 252 * d_data["Mean1"] / norm_data_std_1 * np.sign(d_data["Mean0"])
        color = cmap(int(label)) if label != "Portfolio" else "black"
        plt.scatter(sharpe_0, sharpe_1, label=label, color=color)
    ax.set_autoscale_on(False)
    id_iota = np.linspace(-10, 10, 100)
    plt.plot(id_iota, id_iota, label=None, linestyle="--", color='red')
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.legend()
    plt.ylabel("Sharpe1")
    plt.xlabel("Sharpe0")
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
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import scipy
    # x = np.linspace(-4, 4, 100)
    # y = scipy.stats.norm.cdf(x)
    # plt.plot(x, y)
    # plt.show()

    main()
