"""Analyses to run on the fits."""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm

from .io import get_data, get_fit_path, list_rois, load_fit


def make_table(roi: str, samples: pd.DataFrame, params: list, stats: dict,
               quantiles: list = [0.025, 0.25, 0.5, 0.75, 0.975],
               chain: [int, None] = None) -> pd.DataFrame:
    """Make a table summarizing the fit.

    Args:
        roi (str): A single region, e.g. "US_MI" or "Greece".
        samples (pd.DataFrame): The sample from the fit.
        params (list): The fit parameters to summarize.
        stats (dict): Stats for models computed separately
                      (e.g. from `get_waic_and_loo`)
        quantiles (list, optional): Quantiles to repport.
            Defaults to [0.025, 0.25, 0.5, 0.75, 0.975].
        chain ([type], optional): Optional chain to use. Defaults to None.

    Returns:
        pd.DataFrame: A table of fit parameter summary statistics.
    """
    if chain:
        samples = samples[samples['chain'] == chain]
    dfs = []
    for param in params:
        by_week = False
        if param in samples:
            cols = [param]
        elif '-by-week' in param:
            param = param.replace('-by-week', '')
            cols = [col for col in samples if col.startswith('%s[' % param)]
            by_week = True
        else:
            cols = [col for col in samples if col.startswith('%s[' % param)]
        if not cols:
            print("No param like %s is in the samples dataframe" % param)
        else:
            df = samples[cols]
            if by_week:
                if df.shape[1] >= 7:  # At least one week worth of data
                    # Column 6 will be the last day of the first week
                    # It will contain the average of the first week
                    # Do this every 7 days
                    df = df.T.rolling(7).mean().T.iloc[:, 6::7]
                else:
                    # Just use the last value we have
                    df = df.T.rolling(7).mean().T.iloc[:, -1:]
                    # And then null it because we don't want to trust < 1 week
                    # of data
                    df[:] = None
                df.columns = ['%s (week %d)' % (param, i)
                              for i in range(len(df.columns))]
            try:
                df = df.describe(percentiles=quantiles)
            except ValueError as e:
                print(roi, param, df.shape)
                raise e
            df.index = [float(x.replace('%', ''))/100 if '%' in x else x
                        for x in df.index]
            df = df.drop('count')
            if not by_week:
                # Compute the median across all of the matching column names
                df = df.median(axis=1).to_frame(name=param)
            # Drop the index
            df.columns = [x.split('[')[0] for x in df.columns]
            df.index = pd.MultiIndex.from_product(([roi], df.index),
                                                  names=['roi', 'quantile'])
            dfs.append(df)
    df = pd.concat(dfs, axis=1)
    for stat in ['waic', 'loo', 'lp__rhat']:
        if stat in stats:
            m = stats[stat]
            if m is None:
                m = 0
                s = 0
            else:
                s = stats.get('%s_se' % stat, 0)
            for q in quantiles:
                df.loc[(roi, q), stat] = norm.ppf(q, m, s)
            df.loc[(roi, 'mean'), stat] = m
            df.loc[(roi, 'std'), stat] = s
    for param in params:
        if param not in df:
            df[param] = None
    df = df.sort_index()
    return df


def get_day_labels(data: pd.DataFrame, days: list, t0: int) -> list:
    """Gets labels for days. Used for plotting only.

    Args:
        data (pd.DataFrame): The day-indexed data used for plotting
        days (list): The days for which to get labels.
        t0 (int): The first day of data (case # above a threshold).

    Returns:
        list: Labels to use for those days on the axis of a plot.
    """
    days, day_labels = zip(*enumerate(data.index[t0:]))
    day_labels = ['%s (%d)' % (day_labels[i][:-3], days[i])
                  for i in range(len(days))]
    return day_labels


def get_ss_ifrs(fits_path: str, model_name: str,
                quantiles: list = [0.025, 0.25, 0.5, 0.75, 0.975],
                save: bool = False) -> pd.DataFrame:
    """Gets steady-state Infection Fatality Rates.  Uses an asymptotic equation
    derived from the model which will not match the empirical IFR due both
    right-censoring of deaths and non-linearities.  For reference only.

    Args:
        fits_path (str): Full path to fits directory.
        model_name (str): Model names without the '.stan' suffix.
        quantiles (list, optional): Quantiles to report.
            Defaults to [0.025, 0.25, 0.5, 0.75, 0.975].
        save (bool, optional): Whether to save the results. Defaults to False.

    Returns:
        pd.DataFrame: Regions x Quantiles estimates of steady-state IFR.
    """
    rois = list_rois(fits_path, model_name, 'pkl')
    ifrs = pd.DataFrame(index=rois, columns=quantiles)
    for roi in tqdm(rois):
        fit_path = get_fit_path(fits_path, model_name, roi)
        try:
            fit = load_fit(fit_path, model_name)
        except Exception as e:
            print(e)
        else:
            samples = fit.to_dataframe()
            s = samples
            x = (s['sigmac']/(s['sigmac']+s['sigmau'])) * \
                (s['sigmad']/(s['sigmad']+s['sigmar']))
            ifrs.loc[roi] = x.quantile(quantiles)
    if save:
        ifrs.to_csv(Path(fits_path) / 'ifrs.csv')
    return ifrs


def get_timing(roi: str, data_path: str) -> tuple:
    """[summary]

    Args:
        roi (str): A single region, e.g. "US_MI" or "Greece".
        data_path (str): Full path to the data directory.

    Returns:
        tuple: The first day of data and the first day of mitigation.
    """
    data = get_data(roi, data_path)  # Load the data
    t0date = data[data["new_cases"] >= 1].index[0]
    t0 = data.index.get_loc(t0date)
    try:
        dfm = pd.read_csv(Path(data_path) / 'mitigationprior.csv')\
                .set_index('region')
        tmdate = dfm.loc[roi, 'date']
        tm = data.index.get_loc(tmdate)
    except FileNotFoundError:
        print("No mitigation data found; falling back to default value")
        tm = t0 + 10
        tmdate = data.index[t0]

    print(t0, t0date, tm, tmdate)
    print("t0 = %s (day %d)" % (t0date, t0))
    print("tm = %s (day %d)" % (tmdate, tm))
    return t0, tm


def plot_data_and_fits(data_path: str, roi: str, samples: pd.DataFrame,
                       t0: int, tm: int, chains: [int, None] = None) -> None:
    """Plot the time-series data and the fits together.  Restricted to cases,
    recoveries, and deaths.

    Args:
        data_path (str): Full path to the data directory.
        roi (str): A single region, e.g. "US_MI" or "Greece".
        samples (pd.DataFrame): Samples from the fit.
        t0 (int): Day at which the data begins (threshold # of cases),
        tm (int): Day at which mitigation begins.
        chains ([type], optional): Chain to use. Defaults to None.
    """
    data = get_data(roi, data_path)

    if chains is None:
        chains = samples['chain'].unique()

    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    days = range(data.shape[0])
    days_found = [day for day in days if 'lambda[%d,1]' % (day-t0) in samples]
    days_missing = set(days).difference(days_found)
    print(("Empirical data for days %s is available but fit data for these "
           "day sis missing") % days_missing)
    estimates = {}
    chain_samples = samples[samples['chain'].isin(chains)]

    for i, kind in enumerate(['cases', 'recover', 'deaths']):
        estimates[kind] = [chain_samples['lambda[%d,%d]' % (day-t0, i+1)]
                           .mean() for day in days_found]
        colors = 'bgr'
        cum = data["cum_%s" % kind]
        xticks, xlabels = zip(*[(i, x[:-3]) for i, x in enumerate(cum.index)
                                if i % 2 == 0])

        xlabels = [x[:-3] for i, x in enumerate(cum.index) if i % 2 == 0]
        ax[i, 0].set_title('Cumulative %s' % kind)
        ax[i, 0].plot(cum, 'bo', color=colors[i], label=kind)
        ax[i, 0].axvline(t0, color='k', linestyle="dashed", label='t0')
        ax[i, 0].axvline(tm, color='purple', linestyle="dashed",
                         label='mitigate')
        ax[i, 0].set_xticks(xticks)
        ax[i, 0].set_xticklabels(xlabels, rotation=80, fontsize=8)
        ax[i, 0].legend()

        new = data["new_%s" % kind]
        ax[i, 1].set_title('Daily %s' % kind)
        ax[i, 1].plot(new, 'bo', color=colors[i], label=kind)
        ax[i, 1].axvline(t0, color='k', linestyle="dashed", label='t0')
        ax[i, 1].axvline(tm, color='purple', linestyle="dashed",
                         label='mitigate')
        ax[i, 1].set_xticks(xticks)
        ax[i, 1].set_xticklabels(xlabels, rotation=80, fontsize=8)
        if kind in estimates:
            ax[i, 1].plot(days_found, estimates[kind],
                          label=r'$\hat{%s}$' % kind, linewidth=2, alpha=0.5,
                          color=colors[i])
    ax[i, 1].legend()

    plt.tight_layout()
    fig.suptitle(roi, y=1.02)


def make_histograms(samples: pd.DataFrame, hist_params: list, cols: int = 4,
                    size: int = 3):
    """Make histograms of key parameters.

    Args:
        samples (pd.DataFrame): Samples from the fit.
        hist_params (list): List of parameters from which to make histograms.
        cols (int, optional): Number of columns of plots. Defaults to 3.
        size (int, optional): Overall scale of plots. Defaults to 3.
    """
    cols = min(len(hist_params), cols)
    rows = math.ceil(len(hist_params)/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(size*cols, size*rows))
    chains = samples['chain'].unique()
    for i, param in enumerate(hist_params):
        options = {}
        ax = axes.flat[i]
        for chain in chains:
            if ':' in param:
                param, options = param.split(':')
                options = eval("dict(%s)" % options)
            chain_samples = samples[samples['chain'] == chain][param]
            if options.get('log', False):
                chain_samples = np.log(chain_samples)
            low, high = chain_samples.quantile([0.01, 0.99])
            if high-low < 1e-6:
                low *= 0.99
                high *= 1.01
            bins = np.linspace(low, high, min(25, len(chain_samples)))
            ax.hist(chain_samples, alpha=0.5,
                    bins=bins,
                    label='Chain %d' % chain)
            if options.get('log', False):
                ax.set_xticks(np.linspace(chain_samples.min(),
                                          chain_samples.max(), 5))
                ax.set_xticklabels(['%.2g' % np.exp(x)
                                    for x in ax.get_xticks()])
            ax.set_title(param)
        plt.legend()
    plt.tight_layout()


def make_lineplots(samples: pd.DataFrame, time_params: list, rows: int = 4,
                   cols: int = 4, size: int = 4) -> None:
    """Make line plots smummarizing time-varying parameters.

    Args:
        samples (pd.DataFrame): Samples from the fit.
        time_params (list): List of parameters which vary in time.
        rows (int, optional): Number of rows of plots. Defaults to 4.
        cols (int, optional): Number of columns of plots. Defaults to 4.
        size (int, optional): Overall scale of plots. Defaults to 4.
    """
    cols = min(len(time_params), cols)
    rows = math.ceil(len(time_params)/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(size*cols, size*rows))
    chains = samples['chain'].unique()
    colors = 'rgbk'
    for i, param in enumerate(time_params):
        ax = axes.flat[i]
        for chain in chains:
            cols = [col for col in samples if param in col]
            chain_samples = samples[samples['chain'] == chain][cols]
            quantiles = chain_samples.quantile([0.05, 0.5, 0.95]).T.\
                reset_index(drop=True)
            ax.plot(quantiles.index, quantiles[0.5],
                    label=('Chain %d' % chain), color=colors[chain])
            ax.fill_between(quantiles.index, quantiles[0.05], quantiles[0.95],
                            alpha=0.2, color=colors[chain])
        ax.legend()
        ax.set_title(param)
        ax.set_xlabel('Days')
    plt.tight_layout()
