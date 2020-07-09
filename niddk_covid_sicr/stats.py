"""Compute stats on the results."""

import arviz as az
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from pystan.misc import _summary
from scipy.stats import nbinom
from tqdm.auto import tqdm
from warnings import warn

from .io import extract_samples


def get_rhat(fit) -> float:
    """Get `rhat` for the log-probability of a fit.

    This is a measure of the convergence across sampling chains.
    Good convergence is indicated by a value near 1.0.
    """
    x = _summary(fit, ['lp__'], [])
    summary = pd.DataFrame(x['summary'], columns=x['summary_colnames'], index=x['summary_rownames'])
    return summary.loc['lp__', 'Rhat']


def get_waic(samples: pd.DataFrame) -> dict:
    """Get the Widely-Used Information Criterion (WAIC) for a fit.

    Only use if you don't have arviz (`get_waic_and_loo` is preferred).

    Args:
        samples (pd.DataFrame): Samples extracted from a fit.

    Returns:
        dict: WAIC and se of WAIC for these samples
    """
    from numpy import log, exp, sum, mean, var, sqrt
    # I named the Stan array 'llx'
    ll = samples[[c for c in samples if 'llx' in c]]
    n_samples, n_obs = ll.shape
    # Convert to likelihoods (pray for no numeric precision issues)
    like = exp(ll)
    # log of the mean (across samples) of the likelihood for each observation
    lml = log(mean(like, axis=0))
    # Sum (across observations) of lml
    lppd = sum(lml)
    # Variance (across samples) of the log-likelihood for each observation
    vll = var(ll, axis=0)
    # Sum (across observations) of the vll
    pwaic = sum(vll)
    elpdi = lml - vll
    waic = 2*(-lppd + pwaic)
    # Standar error of the measure
    se = 2*sqrt(n_obs*var(elpdi))
    return {'waic': waic, 'se': se}


def get_waic_and_loo(fit) -> dict:
    warn("`get_waic_and_loo` is deprecated, use `get_fit_quality` instead.",
         DeprecationWarning)
    return get_fit_quality(fit)


def get_fit_quality(fit) -> dict:
    """Compute Widely-Available Information Criterion (WAIC) and
    Leave One Out (LOO) from a fit instance using Arviz.

    Args:
        fit: A PyStan4model instance (i.e. a PyStan fit).

    Returns:
        dict: WAIC and LOO statistics (and se's) for this fit.
    """
    result = {}
    try:
        idata = az.from_pystan(fit, log_likelihood="llx")
    except KeyError as e:
        warn("'%s' not found; waic and loo will not be computed" % str(e),
             stacklevel=2)
        result.update({'waic': 0, 'loo': 0})
    else:
        result.update(dict(az.loo(idata, scale='deviance')))
        result.update(dict(az.waic(idata, scale='deviance')))
    result.update({'lp__rhat': get_rhat(fit)})
    return result


def getllxtensor_singleroi(roi: str, data_path: str, fits_path: str,
                           models_path: str, model_name: str,
                           fit_format: int) -> np.array:
    """Recompute a single log-likelihood tensor (n_samples x n_datapoints).

    Args:
        roi (str): A single ROI, e.g. "US_MI" or "Greece".
        data_path (str): Full path to the data directory.
        fits_path (str): Full path to the fits directory.
        models_path (str): Full path to the models directory.
        model_name (str): The model name (without the '.stan' suffix).
        fit_format (int): The .csv (0) or .pkl (1) fit format.

    Returns:
        np.array: The log-likelihood tensor.
    """
    csv_path = Path(data_path) / ("covidtimeseries_%s_.csv" % roi)
    df = pd.read_csv(csv_path)
    t0 = np.where(df["new_cases"].values > 1)[0][0]
    y = df[['new_cases', 'new_recover', 'new_deaths']].to_numpy()\
        .astype(int)[t0:, :]
    # load samples
    samples = extract_samples(fits_path, models_path, model_name, roi,
                              fit_format)
    S = np.shape(samples['lambda[0,0]'])[0]
    # print(S)
    # get number of observations, check against data above
    for i in range(1000, 0, -1):  # Search for it from latest to earliest
        candidate = '%s[%d,0]' % ('lambda', i)
        if candidate in samples:
            N = i+1  # N observations, add 1 since index starts at 0
            break  # And move on
    print(N)  # run using old data
    print(len(y))
    llx = np.zeros((S, N, 3))
    # # conversion from Stan neg_binom2(n_stan | mu,phi)
    # to scipy.stats.nbinom(k,n_scipy,p)
    # #     n_scipy = phi,    p = phi/mu, k = n_stan
    # t0 = time.time()
    for i in range(S):
        phi = samples['phi'][i]
        for j in range(N):
            mu = max(samples['lambda['+str(j)+',0]'][i], 1)
            llx[i, j, 0] = np.log(nbinom.pmf(max(y[j, 0], 0), phi, phi/mu))
            mu = max(samples['lambda['+str(j)+',1]'][i], 1)
            llx[i, j, 1] = np.log(nbinom.pmf(max(y[j, 1], 0), phi, phi/mu))
            mu = max(samples['lambda['+str(j)+',2]'][i], 1)
            llx[i, j, 2] = np.log(nbinom.pmf(max(y[j, 2], 0), phi, phi/mu))
        print(np.sum(llx[i, :, :]))
        print(samples['ll_'][i])
        print('--')
    return llx


def reweighted_stat(stat_vals: np.array, loo_vals: np.array,
                    loo_se_vals: np.array = None) -> float:
    """Get weighted means of a stat (across models),
    where the weights are related to the LOO's of model/

    Args:
        stat_vals (np.array): Values (across models) of some statistic.
        loo_vals (np.array): Values (across models) of LOO.
        loo_se_vals (np.array, optional): Values (across models) of se of LOO.
            Defaults to None.

    Returns:
        float: A new average value for the statistic, weighted across models.
    """

    # Assume that loo is on a deviance scale (lower is better)
    min_loo = min(loo_vals)
    weights = np.exp(-0.5*(loo_vals-min_loo))
    if loo_se_vals is not None:
        weights *= np.exp(-0.5*loo_se_vals)
    weights = weights/np.sum(weights)
    return np.sum(stat_vals * weights)


def reweighted_stats(raw_table_path: str, save: bool = True,
                     roi_weight='n_data_pts', extra=None, first=None, dates=None) -> pd.DataFrame:
    """Reweight all statistics (across models) according to the LOO
    of each of the models.

    Args:
        raw_table_path (str): Path to the .csv file containing the statistics
                              for each model.
        save (bool, optional): Whether to save the results. Defaults to True.

    Returns:
        pd.DataFrame: The reweighted statistics
                      (i.e. a weighted average across models).
    """
    df = pd.read_csv(raw_table_path, index_col=['model', 'roi', 'quantile'])
    df = df[~df.index.duplicated(keep='last')]
    df.columns.name = 'param'
    df = df.stack('param').unstack(['roi', 'quantile', 'param']).T
    rois = df.index.get_level_values('roi').unique()
    result = pd.Series(index=df.index)
    if first is not None:
        rois = rois[:first]
    for roi in tqdm(rois):
        loo = df.loc[(roi, 'mean', 'loo')]
        loo_se = df.loc[(roi, 'std', 'loo')]
        # An indexer for this ROI
        chunk = df.index.get_level_values('roi') == roi
        result[chunk] = df[chunk].apply(lambda x:
                                        reweighted_stat(x, loo, loo_se),
                                        axis=1)
    result = result.unstack(['param'])
    result = result[~result.index.get_level_values('quantile')
                           .isin(['min', 'max'])]  # Remove min and max
    if extra is not None:
        extra.columns.name = 'param'
        # Don't overwrite with anything already present in the result
        extra = extra[[col for col in extra if col not in result]]
        result = result.join(extra)

    # Add stats for a fixed date
    if dates:
        if isinstance(dates, str):
            dates = [dates]
        for date in dates:
            result = add_fixed_date(result, date, ['Rt', 'car', 'ifr'])

    # Compute global stats
    means = result.unstack('roi').loc['mean'].unstack('param')
    means = means.drop('AA_Global', errors='ignore')
    means = means.drop('US_Region', errors ='ignore')
    means = means[sorted(means.columns)]

    # Get weights for global region and calculate mean and var
    (global_mean, global_var) = get_weight(result, means, roi_weight)

    global_sd = global_var**(1/2)

    result.loc[('AA_Global', 'mean'), :] = global_mean
    result.loc[('AA_Global', 'std'), :] = global_sd



    # Compute super region stats (US states)
    superMeans = means
    superResult = result


    # For Super Region: iterate through index and remove roi's not in super region
    (superMeans, superResult) = filter_region(superResult, superMeans,"US")

    # Get weights for super region and calculate mean and var
    (super_mean, super_var) = get_weight(superResult, superMeans, roi_weight)

    super_sd = super_var**(1/2)

    superResult.loc[('AA_US_Region', 'mean'), :] = super_mean
    superResult.loc[('AA_US_Region', 'std'), :] = super_sd

    # Insert into a new column beside 'R0' the average between super region mean
    #   and the specific ROI
    superResult.insert(1, "superR0_roiR0_Avg", (super_mean[0] + superResult['R0'])/2)

    # Remove superR0 average from non-super region spots
    # So far, this removes it everywhere...
    for i in range(len(superMeans.index)-1):
        if not superResult.index.get_level_values('roi')[i].startswith("US"):
            superResult["superR0_roiR0_Avg"] = np.nan

    superResult = superResult.sort_index()

    if save:
        path = Path(raw_table_path).parent / 'fit_table_reweighted.csv'
        # result.to_csv(path)
        superResult.to_csv(path)
        return superResult

def get_weight(result, means, roi_weight):
    """ Helper function for reweighted_stats() that calculates roi weight for
        either global region or super region.

        Args:
            result (pd.DataFrame): Dataframe that includes global mean (result df)
                                or super mean (superResult)

            means (pd.DataFrame): Global region mean or super region mean

            roi_weight ()

        Returns:
            region_mean: global or super region mean.
            region_var: global or super region variance.
     """
    if roi_weight == 'var':
        inv_var = 1/result.unstack('roi').loc['std']**2
        weights = inv_var.fillna(0).unstack('param')

        region_mean = (means*weights).sum() / weights.sum()
        region_var = ((weights*((means - region_mean)**2)).sum()/weights.sum())

    elif roi_weight == 'waic':
        waic = means['waic']
        n_data = means['n_data_pts']
        # Assume that waic is on a deviance scale (lower is better)
        weights = np.exp(-0.5*waic/n_data)
        region_mean = means.mul(weights, axis=0).sum() / weights.sum()
        region_var = (((means - region_mean)**2).mul(weights, axis=0)).sum()/weights.sum()

    elif roi_weight == 'n_data_pts':
        n_data = means['n_data_pts']
        # Assume that waic is on a deviance scale (lower is better)
        weights = n_data
        region_mean = means.mul(weights, axis=0).sum() / weights.sum()
        region_var = (((means - region_mean)**2).mul(weights, axis=0)).sum()/weights.sum()

    return region_mean, region_var
#
def filter_region(superResult, superMeans, region):
    """ Helper function for reweighted_stats() that filters rois based on the
    defined super region and drops non-super region rois from the DataFrame.

    Args:
    Returns:
        superMeans: DataFrame containing
    """
    for i in range(len(superMeans.index)-1):
        if not superMeans.index[i].startswith(region):
            superMeans = superMeans.drop(superMeans.index[i])
            superResult = superResult.drop(superResult.index[i])
    return superMeans, superResult
    #     """ Helper function for reweighted_stats() that filters .
    #         Args:
    #             superResult (pd.DataFrame): Copy of result DataFrame
    #             superMeans (pd.DataFrame): Copy of means DataFrame
    #         Returns:
    #             superMeans, superResult: DataFrames on
    #      """

def days_into_2020(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    one_one = datetime.strptime('2020-01-01', '%Y-%m-%d')
    return (date - one_one).days


def get_roi_week(date_str, roi_day_one):
    days = days_into_2020(date_str)
    roi_days = days - roi_day_one
    try:
        roi_week = int(roi_days/7)
    except:
        roi_week = 9999
    return roi_week


def add_fixed_date(df, date_str, stats):
    for roi in df.index:
        week = get_roi_week(date_str, df.loc[roi, 't0'])
        for stat in stats:
            col = '%s (week %d)' % (stat, week)
            new_col = '%s (%s)' % (stat, date_str)
            if col in df:
                df.loc[roi, new_col] = df.loc[roi, col]
            else:
                df.loc[roi, new_col] = None
    return df
