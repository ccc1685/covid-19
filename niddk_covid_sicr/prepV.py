from datetime import datetime
import numpy as np
from numpy.random import gamma, exponential, lognormal,normal
import pandas as pd
from pathlib import Path
import sys

import niddk_covid_sicr as ncs


def get_stan_dataV(full_data_path, args):
    df = pd.read_csv(full_data_path)
    if getattr(args, 'last_date', None):
        try:
            datetime.strptime(args.last_date, '%m/%d/%y')
        except ValueError:
            msg = "Incorrect --last-date format, should be MM/DD/YY"
            raise ValueError(msg)
        else:
            df = df[df['dates2'] <= args.last_date]

    # t0 := where to start time series, index space
    try:
        t0 = np.where(df["new_cases"].values >= 5)[0][0]
    except IndexError:
        return [None, None]
    # tm := start of mitigation, index space

    try:
        dfm = pd.read_csv(args.data_path / 'mitigationprior.csv')
        tmdate = dfm.loc[dfm.region == args.roi, 'date'].values[0]
        tm = np.where(df["dates2"] == tmdate)[0][0]
    except Exception:
        print("Could not use mitigation prior data; setting mitigation prior to default.")
        tm = t0 + 10

    n_proj = 120
    stan_data = {}
    stan_data['n_ostates'] = 5
    stan_data['tm'] = tm
    stan_data['ts'] = np.arange(t0, len(df['dates2']) + n_proj)
    stan_data['y'] = df[['new_cases', 'new_recover', 'new_deaths', 'new_hosp', 'new_icu']].to_numpy()\
        .astype(int)[t0:, :]
    stan_data['n_obs'] = len(df['dates2']) - t0
    stan_data['n_total'] = len(df['dates2']) - t0 + n_proj
    if args.fixed_t:
        global_start = datetime.strptime('01/22/20', '%m/%d/%y')
        frame_start = datetime.strptime(df['dates2'][0], '%m/%d/%y')
        offset = (frame_start - global_start).days
        stan_data['tm'] += offset
        stan_data['ts'] += offset
    return stan_data, df['dates2'][t0]


def get_n_data(stan_data):
    if stan_data:
        return (stan_data['y'] > 0).ravel().sum()
    else:
        return 0


# functions used to initialize parameters
def get_init_funV(args, stan_data, force_fresh=False):
    if args.init and not force_fresh:
        try:
            init_path = Path(args.fits_path) / args.init
            model_path = Path(args.models_path) / args.model_name
            result = ncs.last_sample_as_dict(init_path, model_path)
        except Exception:
            print("Couldn't use last sample from previous fit to initialize")
            return init_fun(force_fresh=True)
        else:
            print("Using last sample from previous fit to initialize")
    else:
        print("Using default values to initialize fit")

        result = {'f1': 1.75,
                  'f2': .25,
                  'sigmaVS': exponential(1/10.),
                  'sigmaSR':  exponential(1/400.),
                  'sigmaSRI':  exponential(1/400.),
                  'sigmaIV':  exponential(1/100.),
                  'sigmaRI':  .01,
                  'sigmaDI':  .03,
                  'sigmaRC':  .08,
                  'sigmaDC':  .003,
                  'sigmaRH1':  1.,
                  'sigmaRH2':  .1,
                  'sigmaRV':  exponential(1/10.),
                  'sigmaH1C':  .01,
                  'sigmaH1V':  exponential(1/10.),
                  'sigmaH2H1':  .4,
                  'sigmaDH1':  .003,
                  'sigmaDH2':  .001,
                  'sigmaM':  exponential(1/10.),
                  'mbase':  exponential(1/10.),
                  'q': exponential(.5),
                  'n_pop': normal(4e6, 1e4)
                          }
        # result = {'f1': exponential(1.),
        #           'f2': exponential(1.),
        #           'sigmaVS': exponential(1/10.),
        #           'sigmaSR':  exponential(1/400.),
        #           'sigmaSRI':  exponential(1/400.),
        #           'sigmaIV':  exponential(1/100.),
        #           'sigmaRI':  exponential(1/10.),
        #           'sigmaDI':  exponential(1/30.),
        #           'sigmaRC':  exponential(1/10.),
        #           'sigmaDC':  exponential(1/10.),
        #           'sigmaRH1':  exponential(1/10.),
        #           'sigmaRH2':  exponential(1/10.),
        #           'sigmaRV':  exponential(1/10.),
        #           'sigmaH1C':  exponential(1/10.),
        #           'sigmaH1V':  exponential(1/10.),
        #           'sigmaH2H1':  exponential(1/10.),
        #           'sigmaRH1':  exponential(1/10.),
        #           'sigmaDH1':  exponential(1/10.),
        #           'sigmaDH2':  exponential(1/10.),
        #           'sigmaM':  exponential(1/10.),
        #           'mbase':  exponential(1/10.),
        #           'q': exponential(.1),
        #           'n_pop': normal(1e6, 1e4)
        #                   }

    def init_fun():
        return result
    return init_fun
