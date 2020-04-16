import argparse
import numpy as np
import os
from pathlib import Path
import pickle
import pandas as pd
import pystan
import sys
from __init__ import load_or_compile_stan_model

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description='Runs all the Stan models')

parser.add_argument('model_name',
                    help='Name of the Stan model file (without extension)')
parser.add_argument('-dp', '--data_path', default='../data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits_path', default='./fits',
                    help='Path to directory to save fit files')
parser.add_argument('-r', '--roi', default='US_NY',
                    help='ROI to use')
parser.add_argument('-ch', '--n_chains', default=4,
                    help='Number of chains to run')
parser.add_argument('-wm', '--n_warmups', default=500,
                    help='Number of chains to run')
parser.add_argument('-it', '--n_iter', default=1000,
                    help='Number of chains to run')
parser.add_argument('-tn', '--n_thin', default=1,
                    help='Number of chains to run')
parser.add_argument('-th', '--n_threads', default=0,
                    help='Number of threads to use the whole run')
parser.add_argument('-ad', '--adapt_delta', default=0.995,
                    help='Adapt delta control parameter')
args = parser.parse_args()
if args.n_threads == 0:
    args.n_threads = args.n_chains


csv = Path(args.data_path) / ("covid_timeseries_%s.csv" % args.roi)
csv = csv.resolve()
assert csv.exists(), "No such csv file: %s" % csv

control = {'adapt_delta': args.adapt_delta}
stanrunmodel = load_or_compile_stan_model(args.model_name, force_recompile=False)
df = pd.read_csv(csv)

# t0 := where to start time series, index space
t0 = np.where(df["new_cases"].values>1)[0][0]
# tm := start of mitigation, index space
tm = t0 + 10

stan_data = {}
stan_data['n_scale'] = 1000 #use this instead of population
stan_data['n_theta'] = 9
stan_data['n_difeq'] = 5
stan_data['n_ostates'] = 3
stan_data['t0'] = t0-1 #to for ODE is one day, index before start of series
stan_data['tm'] = tm
stan_data['ts'] = np.arange(t0,len(df['dates2']))
stan_data['y'] = (df[['new_cases','new_recover','new_deaths']].to_numpy()).astype(int)[t0:,:]
stan_data['n_obs'] = len(df['dates2']) - t0

# function used to initialize parameters
def init_fun():
        x = {'theta':
               [np.random.gamma(1.5,1/1.5)]
             + [np.random.gamma(1.5,1/4.5)]
             + [np.random.gamma(2.,.1/2)]
             + [np.random.gamma(2.,.1/2)]
             + [np.random.gamma(2.,.1/2)]
             + [np.random.exponential(2.)]
             + [np.random.exponential(3.)]
             + [np.random.lognormal(np.log(stan_data['tm']),.5)]
             + [np.random.exponential(1.)]
            }
        return x

# Fit Stan
fit = stanrunmodel.sampling(data=stan_data, init=init_fun, control=control, chains=args.n_chains, chain_id=np.arange(args.n_chains),
                            warmup=args.n_warmups, iter=args.n_iter, thin=args.n_thin)

# Uncomment to print fit summary
print(fit)

# Save fit
save_dir = Path(args.fits_path)
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / ("%s_%s.csv" % (args.model_name, args.roi))
result = fit.to_dataframe().to_csv(save_path)