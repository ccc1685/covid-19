"""Functions for getting data needed to fit the models."""

import bs4
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from typing import Union
from urllib.error import HTTPError
import urllib.request, json

JHU_FILTER_DEFAULTS = {'confirmed': 5, 'recovered': 1, 'deaths': 0}
COVIDTRACKER_FILTER_DEFAULTS = {'cum_cases': 5, 'cum_recover': 1, 'cum_deaths': 0}

def get_jhu(data_path: str, filter_: Union[dict, bool] = True) -> None:
    """Gets data from Johns Hopkins CSSEGIS (countries only).

    https://coronavirus.jhu.edu/map.html
    https://github.com/CSSEGISandData/COVID-19

    Args:
        data_path (str): Full path to data directory.

    Returns:
        None
    """
    # Where JHU stores their data
    url_template = ("https://raw.githubusercontent.com/CSSEGISandData/"
                    "COVID-19/master/csse_covid_19_data/"
                    "csse_covid_19_time_series/time_series_covid19_%s_%s.csv")

    # Scrape the data
    dfs = {}
    for region in ['global', 'US']:
        dfs[region] = {}
        for kind in ['confirmed', 'deaths', 'recovered']:
            url = url_template % (kind, region)  # Create the full data URL
            try:
                df = pd.read_csv(url)  # Download the data into a dataframe
            except HTTPError:
                print("Could not download data for %s, %s" % (kind, region))
            else:
                if region == 'global':
                    has_no_province = df['Province/State'].isnull()
                    # Whole countries only; use country name as index
                    df1 = df[has_no_province].set_index('Country/Region')
                    more_dfs = []
                    for country in ['China', 'Canada', 'Australia']:
                        if country == 'Canada' and kind in 'recovered':
                            continue
                        is_c = df['Country/Region'] == country
                        df2 = df[is_c].sum(axis=0, skipna=False).to_frame().T
                        df2['Country/Region'] = country
                        df2 = df2.set_index('Country/Region')
                        more_dfs.append(df2)
                    df = pd.concat([df1] + more_dfs)
                elif region == 'US':
                    # Use state name as index
                    df = df.set_index('Province_State')
                df = df[[x for x in df if any(year in x for year in ['20', '21'])]]  # Use only data columns
                                                # 20 or 21 signifies 2020 or 2021
                dfs[region][kind] = df  # Add to dictionary of dataframes

    # Generate a list of countries that have "good" data,
    # according to these criteria:
    good_countries = get_countries(dfs['global'], filter_=filter_)

    # For each "good" country,
    # reformat and save that data in its own .csv file.
    source = dfs['global']
    for country in tqdm(good_countries, desc='Countries'):  # For each country
        # If we have data in the downloaded JHU files for that country
        if country in source['confirmed'].index:
            df = pd.DataFrame(columns=['dates2', 'cum_cases', 'cum_deaths',
                                       'cum_recover', 'new_cases',
                                       'new_deaths', 'new_recover',
                                       'new_uninfected'])
            df['dates2'] = source['confirmed'].columns
            df['dates2'] = df['dates2'].apply(fix_jhu_dates)
            df['cum_cases'] = source['confirmed'].loc[country].values
            df['cum_deaths'] = source['deaths'].loc[country].values
            df['cum_recover'] = source['recovered'].loc[country].values
            df[['new_cases', 'new_deaths', 'new_recover']] = \
                df[['cum_cases', 'cum_deaths', 'cum_recover']].diff()
            df['new_uninfected'] = df['new_recover'] + df['new_deaths']
            # Fill NaN with 0 and convert to int
            dfs[country] = df.set_index('dates2').fillna(0).astype(int)
            # Overwrite old data
            dfs[country].to_csv(data_path /
                                ('covidtimeseries_%s.csv' % country))
        else:
            print("No data for %s" % country)

def fix_jhu_dates(x):
    y = datetime.strptime(x, '%m/%d/%y')
    return datetime.strftime(y, '%m/%d/%y')


def fix_ct_dates(x):
    return datetime.strptime(str(x), '%Y%m%d')


def get_countries(d: pd.DataFrame, filter_: Union[dict, bool] = True):
    """Get a list of countries from a global dataframe optionally passing
    a quality check

    Args:
        d (pd.DataFrame): Data from JHU tracker (e.g. df['global]).
        filter (bool, optional): Whether to filter by quality criteria.
    """
    good = set(d['confirmed'].index)
    if filter_ and not isinstance(filter_, dict):
        filter_ = JHU_FILTER_DEFAULTS
    if filter_:
        for key, minimum in filter_.items():
            enough = d[key].index[d[key].max(axis=1) >= minimum].tolist()
            good = good.intersection(enough)
    bad = set(d['confirmed'].index).difference(good)
    print("JHU data acceptable for %s" % ','.join(good))
    print("JHU data not acceptable for %s" % ','.join(bad))
    return good


def get_covid_tracking(data_path: str, filter_: Union[dict, bool] = True,
                       fixes: bool = False) -> None:
    """Gets data from The COVID Tracking Project (US states only).

    https://covidtracking.com

    Args:
        data_path (str): Full path to data directory.

    Returns:
        None
    """
    url = ("https://raw.githubusercontent.com/COVID19Tracking/"
           "covid-tracking-data/master/data/states_daily_4pm_et.csv")
    df_raw = pd.read_csv(url)

    states = df_raw['state'].unique()

    if filter_ and not isinstance(filter_, dict):
        filter_ = COVIDTRACKER_FILTER_DEFAULTS
    good = []
    bad = []
    for state in tqdm(states, desc='US States'):  # For each country
        source = df_raw[df_raw['state'] == state]  # Only the given state
        # If we have data in the downloaded file for that state
        df = pd.DataFrame(columns=['dates2', 'cum_cases', 'cum_deaths',
                                       'cum_recover', 'new_cases',
                                       'new_deaths', 'new_recover',
                                       'new_uninfected'])

        # Convert date format
        df['dates2'] = source['date'].apply(fix_ct_dates)
        df['cum_cases'] = source['positive'].values
        df['cum_deaths'] = source['death'].values
        df['cum_recover'] = source['recovered'].values
        # Fill NaN with 0 and convert to int
        df.sort_values(by=['dates2'], inplace=True) # sort by datetime obj before converting to string
        df['dates2'] = pd.to_datetime(df['dates2']).dt.strftime('%m/%d/%y') # convert dates to string
        df = df.set_index('dates2').fillna(0).astype(int)
        enough = True
        if filter_:
            for key, minimum in filter_.items():
                enough *= (df[key].max() >= minimum)
        if not enough:
            bad.append(state)
        else:
            df[['new_cases', 'new_deaths', 'new_recover']] = \
            df[['cum_cases', 'cum_deaths', 'cum_recover']].diff()
            df['new_uninfected'] = df['new_recover'] + df['new_deaths']
            df = df.fillna(0).astype(int)
            # Overwrite old data
            df.to_csv(data_path / ('covidtimeseries_US_%s.csv' % state))
            good.append(state)
    print("COVID Tracking data acceptable for %s" % ','.join(good))
    print("COVID Tracking data not acceptable for %s" % ','.join(bad))

def get_canada(data_path: str, filter_: Union[dict, bool] = True,
                       fixes: bool = False) -> None:
    """ Gets data from Canada's Open Covid group for Canadian Provinces.
        https://opencovid.ca/
    """
    dfs = [] # we will append dfs for cases, deaths, recovered here
    # URL for API call to get Province-level timeseries data starting on Jan 22 2020
    url_template = 'https://api.opencovid.ca/timeseries?stat=%s&loc=prov&date=01-22-2020'
    for kind in ['cases', 'mortality', 'recovered']:
        url_path = url_template % kind  # Create the full data URL
        with urllib.request.urlopen(url_path) as url:
            data = json.loads(url.read().decode())
            source = pd.json_normalize(data[kind])
            if kind == 'cases':
                source.drop('cases', axis=1, inplace=True) # removing this column so
                # we can index into date on all 3 dfs at same position
            source.rename(columns={source.columns[1]: "date" }, inplace=True)
            dfs.append(source)
    cases = dfs[0]
    deaths = dfs[1]
    recovered = dfs[2]
    # combine dfs
    df_rawtemp = cases.merge(recovered, on=['date', 'province'], how='outer')
    df_raw = df_rawtemp.merge(deaths, on=['date', 'province'], how='outer')
    df_raw.fillna(0, inplace=True)

    provinces = ['Alberta', 'BC', 'Manitoba', 'New Brunswick', 'NL',
                'Nova Scotia', 'Nunavut', 'NWT', 'Ontario', 'PEI', 'Quebec',
                'Saskatchewan', 'Yukon']

    # Export timeseries data for each province
    for province in tqdm(provinces, desc='Canadian Provinces'):
        source = df_raw[df_raw['province'] == province]  # Only the given province
        df = pd.DataFrame(columns=['dates2','cum_cases', 'cum_deaths',
                                   'cum_recover', 'new_cases',
                                   'new_deaths', 'new_recover',
                                   'new_uninfected'])
        df['dates2'] = source['date'].apply(fix_canada_dates) # Convert date format
        df['cum_cases'] = source['cumulative_cases'].values
        df['cum_deaths'] = source['cumulative_deaths'].values
        df['cum_recover'] = source['cumulative_recovered'].values

        df[['new_cases', 'new_deaths', 'new_recover']] = \
            df[['cum_cases', 'cum_deaths', 'cum_recover']].diff()
        df['new_uninfected'] = df['new_recover'] + df['new_deaths']
        df.sort_values(by=['dates2'], inplace=True) # sort by datetime obj before converting to string
        df['dates2'] = pd.to_datetime(df['dates2']).dt.strftime('%m/%d/%y') # convert dates to string
        df = df.set_index('dates2').fillna(0).astype(int) # Fill NaN with 0 and convert to int
        df.to_csv(data_path / ('covidtimeseries_CA_%s.csv' % province))

def fix_canada_dates(x):
    return datetime.strptime(x, '%d-%m-%Y')

def fix_negatives(data_path: str, plot: bool = False) -> None:
    """Fix negative values in daily data.

    The purpose of this script is to fix spurious negative values in new daily
    numbers.  For example, the cumulative total of cases should not go from N
    to a value less than N on a subsequent day.  This script fixes this by
    nulling such data and applying a monotonic spline interpolation in between
    valid days of data.  This only affects a small number of regions.  It
    overwrites the original .csv files produced by the functions above.

    Args:
        data_path (str): Full path to data directory.
        plot (bool): Whether to plot the changes.

    Returns:
        None
    """
    csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
    for csv in tqdm(csvs, desc="Regions"):
        roi = str(csv).split('.')[0].split('_')[-1]
        df = pd.read_csv(csv)
        # Exclude final day because it is often a partial count.
        df = df.iloc[:-1]
        df = fix_neg(df, roi, plot=plot)
        df.to_csv(data_path / (csv.name.split('.')[0]+'.csv'))


def fix_neg(df: pd.DataFrame, roi: str,
            columns: list = ['cases', 'deaths', 'recover'],
            plot: bool = False) -> pd.DataFrame:
    """Used by `fix_negatives` to fix negatives values for a single region.

    This function uses monotonic spline interpolation to make sure that
    cumulative counts are non-decreasing.

    Args:
        df (pd.DataFrame): DataFrame containing data for one region.
        roi (str): One region, e.g 'US_MI' or 'Greece'.
        columns (list, optional): Columns to make non-decreasing.
            Defaults to ['cases', 'deaths', 'recover'].
    Returns:
        pd.DataFrame: [description]
    """
    for c in columns:
        cum = 'cum_%s' % c
        new = 'new_%s' % c
        before = df[cum].copy()
        non_zeros = df[df[new] > 0].index
        has_negs = before.diff().min() < 0
        if len(non_zeros) and has_negs:
            first_non_zero = non_zeros[0]
            maxx = df.loc[first_non_zero, cum].max()
            # Find the bad entries and null the corresponding
            # cumulative column, which are:
            # 1) Cumulative columns which are zero after previously
            # being non-zero
            bad = df.loc[first_non_zero:, cum] == 0
            df.loc[bad[bad].index, cum] = None
            # 2) New daily columns which are negative
            bad = df.loc[first_non_zero:, new] < 0
            df.loc[bad[bad].index, cum] = None
            # Protect against 0 null final value which screws up interpolator
            if np.isnan(df.loc[df.index[-1], cum]):
                df.loc[df.index[-1], cum] = maxx
            # Then run a loop which:
            while True:
                # Interpolates the cumulative column nulls to have
                # monotonic growth
                after = df[cum].interpolate('pchip')
                diff = after.diff()
                if diff.min() < 0:
                    # If there are still negative first-differences at this
                    # point, increase the corresponding cumulative values by 1.
                    neg_index = diff[diff < 0].index
                    df.loc[neg_index, cum] += 1
                else:
                    break
                # Then repeat
            if plot:
                plt.figure()
                plt.plot(df.index, before, label='raw')
                plt.plot(df.index, after, label='fixed')
                r = np.corrcoef(before, after)[0, 1]
                plt.title("%s %s Raw vs Fixed R=%.5g" % (roi, c, r))
                plt.legend()
        else:
            after = before
        # Make sure the first differences are now all non-negative
        assert after.diff().min() >= 0
        # Replace the values
        df[new] = df[cum].diff().fillna(0).astype(int).values
    return df


def negify_missing(data_path: str) -> None:
    """Fix negative values in daily data.

    The purpose of this script is to fix spurious negative values in new daily
    numbers.  For example, the cumulative total of cases should not go from N
    to a value less than N on a subsequent day.  This script fixes this by
    nulling such data and applying a monotonic spline interpolation in between
    valid days of data.  This only affects a small number of regions.  It
    overwrites the original .csv files produced by the functions above.

    Args:
        data_path (str): Full path to data directory.
        plot (bool): Whether to plot the changes.

    Returns:
        None
    """
    csvs = [x for x in data_path.iterdir() if 'covidtimeseries' in str(x)]
    for csv in tqdm(csvs, desc="Regions"):
        roi = str(csv).split('.')[0].split('_')[-1]
        df = pd.read_csv(csv)
        for kind in ['cases', 'deaths', 'recover']:
            if df['cum_%s' % kind].sum() == 0:
                print("Negifying 'new_%s' for %s" % (kind, roi))
                df['new_%s' % kind] = -1
        out = data_path / (csv.name.split('.')[0]+'.csv')
        df.to_csv(out)
