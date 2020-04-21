import argparse
from importlib import import_module
import numpy as np
import os
from pathlib import Path
import pickle
import pandas as pd
import pystan
import sys

# Make sure the directory containing 'Stan' is on the path
path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(path))

from Stan import load_or_compile_stan_model

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
parser.add_argument('-ch', '--n_chains', type=int, default=4,
                    help='Number of chains to run')
parser.add_argument('-wm', '--n_warmups', type=int, default=500,
                    help='Number of warmups')
parser.add_argument('-it', '--n_iter', type=int, default=1000,
                    help='Number of iterations')
parser.add_argument('-tn', '--n_thin', type=int, default=1,
                    help='thinning number')
parser.add_argument('-th', '--n_threads', type=int, default=0,
                    help='Number of threads to use the whole run')
parser.add_argument('-ad', '--adapt_delta', type=float, default=0.995,
                    help='Adapt delta control parameter')
parser.add_argument('-f', '--fit_format', type=int, default=1,
                    help='Version of fit format')
args = parser.parse_args()
if args.n_threads == 0:
    args.n_threads = args.n_chains

csv = Path(args.data_path) / ("covidtimeseries_%s.csv" % args.roi)
csv = csv.resolve()
assert csv.exists(), "No such csv file: %s" % csv


m = import_module('Stan.%s' % args.model_name)

control = {'adapt_delta': args.adapt_delta}
stanrunmodel = load_or_compile_stan_model(args.model_name, force_recompile=False)
df = pd.read_csv(csv)

# t0 := where to start time series, index space
t0 = np.where(df["new_cases"].values>1)[0][0]
# tm := start of mitigation, index space

try:
    dfm = pd.read_csv(args.data_path / 'mitigationprior.csv')
    tmdate = dfm.loc[dfm.region==args.roi, 'date'].values[0]
    tm = np.where(df["dates2"]==tmdate)[0][0]
except:
    tm = t0 + 10

stan_data = {}
stan_data['n_scale'] = 1000 #use this instead of population
# stan_data['n_theta'] = 8
stan_data['n_difeq'] = 5
stan_data['n_ostates'] = 3
stan_data['t0'] = t0-1 #to for ODE is day one, index before start of series
stan_data['tm'] = tm
stan_data['ts'] = np.arange(t0,len(df['dates2']))
stan_data['y'] = (df[['new_cases','new_recover','new_deaths']].to_numpy()).astype(int)[t0:,:]
stan_data['n_obs'] = len(df['dates2']) - t0

init = [m.init_func(stan_data)] * args.n_chains

# Fit Stan
fit = stanrunmodel.sampling(data=stan_data, init=init, control=control, chains=args.n_chains, chain_id=np.arange(args.n_chains),
                            warmup=args.n_warmups, iter=args.n_iter, thin=args.n_thin)

# Uncomment to print fit summary
print(fit)

# Save fit
save_dir = Path(args.fits_path)
save_dir.mkdir(parents=True, exist_ok=True)
if args.fit_format == 0:
    save_path = save_dir / ("%s_%s.csv" % (args.model_name, args.roi))
    result = fit.to_dataframe().to_csv(save_path)
else:
    save_path = save_dir / ("%s_%s.pkl" % (args.model_name, args.roi))
    with open(save_path, "wb") as f:
        pickle.dump({'model_name' : args.model_name, 'model_code': stanrunmodel.model_code, 'fit' : fit},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
