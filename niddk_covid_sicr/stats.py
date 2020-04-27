"""Compute stats on the results."""

import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import nbinom

from .io import extract_samples


def get_waic(samples: pd.DataFrame) -> dict:
    """Extract all the observation-wise log-likelihoods from the samples
    dataframe.

    Only use if you don't have arviz.

    Args:
        samples (pd.DataFrame): [description]

    Returns:
        dict: [description]
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
    """Compute WAIC and LOO from a fit instance using Arviz.

    Args:
        fit ([type]): [description]

    Returns:
        dict: [description]
    """
    idata = az.from_pystan(fit, log_likelihood="llx")
    result = {}
    result.update(dict(az.loo(idata, scale='deviance')))
    result.update(dict(az.waic(idata, scale='deviance')))
    return result


def getllxtensor_singleroi(roi: str, datapath: str, fits_path: str,
                           models_path: str, model_name: str,
                           fit_format: int) -> np.array:
    """[summary]

    Args:
        roi (str): [description]
        datapath (str): [description]
        fits_path (str): [description]
        models_path (str): [description]
        model_name (str): [description]
        fit_format (int): [description]

    Returns:
        np.array: [description]
    """
    csv = datapath + "covidtimeseries_%s_.csv" % roi
    df = pd.read_csv(csv)
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
        stat_vals (np.array): [description]
        loo_vals (np.array): [description]
        loo_se_vals (np.array, optional): [description]. Defaults to None.

    Returns:
        float: [description]
    """
    min_loo = min(loo_vals)
    weights = np.exp(-0.5*(loo_vals-min_loo))
    if loo_se_vals is not None:
        weights *= np.exp(-0.5*loo_se_vals)
    weights = weights/np.sum(weights)
    return np.sum(stat_vals * weights)


def reweighted_stats(raw_table_path: str, save: bool = True) -> pd.DataFrame:
    """[summary]

    Args:
        raw_table_path (str): [description]
        save (bool, optional): [description]. Defaults to True.

    Returns:
        pd.DataFrame: [description]
    """
    df = pd.read_csv(raw_table_path, index_col=['model', 'roi', 'quantile'])
    df.columns.name = 'param'
    df = df.stack('param').unstack(['roi', 'quantile', 'param']).T
    rois = df.index.get_level_values('roi').unique()
    result = pd.Series(index=df.index)
    for roi in rois:
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

    # Compute global stats
    means = result.unstack('roi').loc['mean'].unstack('param')
    inv_var = 1/result.unstack('roi').loc['std']**2
    weights = inv_var.fillna(0).unstack('param')
    weights.loc['AA_Global'] = 0  # Don't include the global numbers yet
    global_mean = (means*weights).sum() / weights.sum()
    global_var = ((weights*((means - global_mean)**2)).sum()/weights.sum())
    global_sd = global_var**(1/2)
    result.loc[('AA_Global', 'mean'), :] = global_mean
    result.loc[('AA_Global', 'std'), :] = global_sd
    result.loc[('AA_Global', 'std'), :] = global_sd
    result = result.sort_index()

    if save:
        path = Path(raw_table_path).parent / 'fit_table_reweighted.csv'
        result.to_csv(path)
    return result
