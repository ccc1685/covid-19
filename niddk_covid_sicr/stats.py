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


def add_ir(data_path:str, tables_path:str):
    """ Add infectivity ratio IR to tables using regional populations to compute
    the fraction of population infected.

    IR = cases * CAR / population
    Arguments:
        data_path(str): Data-path to timeseries files for getting cum_case counts
                        per roi.
        tables_path(str): Tables-path to tables for adding IR calculation to existing
                          tables.
    Returns:
        None.
    """
    data_path = Path(data_path)
    tables_path = Path(tables_path)
    try:
        cases_csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
    except:
        print("Missing timeseries files in data-path -- run scripts/get-data.py")
    cases_data = []

    for csv in cases_csvs:
        roi = str(csv).split('.')[0].split('_') # get roi name
        if len(roi) > 2: # handle US_ and CA_ prefixes
            roi = roi[1] + '_' + roi[2]
        else: # if not US state or Canadian province
            roi = roi[1]
        df_cases = pd.read_csv(csv)
        cases_dict = {}
        cum_cases = df_cases['cum_cases'][df_cases.index[-1]] # get most recent cum case count
        cases_dict['roi'] = roi
        cases_dict['cum_cases'] = cum_cases
        cases_data.append(cases_dict)

    df_cases = pd.DataFrame(cases_data)
    df_cases.set_index('roi')

    table_csvs = [x for x in tables_path.iterdir() if 'table.csv' in str(x)]
    try:
        df_pop = pd.read_csv(data_path / 'population_estimates.csv') # get population counts
    except:
        print("Missing population_estimates.csv in data-path")
    df_pop.drop(columns=['Sources'], inplace=True)

    for table in table_csvs:
        df = pd.read_csv(table)
        col = ['population', 'cum_cases', 'ir'] # if exits, drop these columns to avoid duplicates
        for i in col:
            if i in df.columns:
                df.drop(columns=[i], inplace=True)
        df = df.merge(df_pop, on='roi', how='left')
        df = df.merge(df_cases, on='roi', how='left')
        df['ir'] = (df['cum_cases'] * df['car']) / df['population']
        df.to_csv(table, index=False)


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
    result = result.sort_index()

    # Compute stats for a superregion (Asia, Southern Asia, United States, etc)
    super_means = means
    super_result = result

    # Define superregion as second argument and iterate through index removing
    #   rois not in superregion.
    (super_means, region) = filter_region(super_means, 'United States')


    # Get weights for superregion and calculate mean and variance.
    (super_mean, super_var) = get_weight(super_result, super_means, roi_weight)

    super_sd = super_var**(1/2)

    super_result.loc[('AA_'+region, 'mean'), :] = super_mean
    super_result.loc[('AA_'+region, 'std'), :] = super_sd

    # Insert into a new column beside 'R0' the average between superregion mean
    #   and ROI in that row.
    super_result.insert(1, region+"_avg", (super_mean[0] + super_result['R0'])/2)

    super_result = super_result.sort_index()

    if save:
        path = Path(raw_table_path).parent / 'fit_table_reweighted.csv'
        super_result.to_csv(path)
        return super_result

def get_weight(result, means, roi_weight):
    """ Helper function for reweighted_stats() that calculates roi weight for
        either global region or superregion.

        Args:
            result (pd.DataFrame): Dataframe that includes global mean (result df)
                                or super mean (super_result)

            means (pd.DataFrame): Global region mean or superregion mean

            roi_weight (str): argument referenced in reweighted_stats()

        Returns:
            region_mean: global or superregion mean.
            region_var: global or superregion variance.
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
def filter_region(super_means, region):
    """ Helper function for reweighted_stats() that filters rois based on the
    defined superregion and drops non-superregion rois from the DataFrame that
    gets used to calculate superregion mean and variance.

    Args:
        super_means (pd.DataFrame): DataFrame containing all ROI means.
        region (str): superregion in question; can be a region or subregion
                      (Europe, Northern Europe, etc).

    Returns:
        super_means (pd.DataFrame): DataFrame containing means for ROIs that fall
                                   under superregion.
        region (str): superregion in question; return value is used to create
                      column and index names.
    """
    # Open CSV containing rois, regions, and subregions.
    super_region = pd.read_csv('niddk_covid_sicr/rois.csv')

    # Find all rois that fall under specified region.
    super_region = super_region[(super_region['subregion']==region) | (super_region['region']==region)]

    # If roi not in superregion list of rois, exclude from superregion stats
    #   calculations by dropping from DataFrame
    for i in super_means.index:
        if i not in super_region.values:
            super_means = super_means.drop(index=i)

    return super_means, region

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
