import numpy as np
import os
from pathlib import Path
import pickle
import pandas as pd
import pystan
import sys
from __init__ import load_or_compile_stan_model

model_name = sys.argv[1]
roi = sys.argv[2]
n_jobs = sys.argv[3]

csv = Path(__file__).parent / ".." / "data" / ("covid_timeseries_%s.csv" % roi)
csv = csv.resolve()
assert csv.exists(), "No such csv file: %s" % csv

n_chains=4
n_warmups=5000
n_iter=10000
n_thin=1
control = {'adapt_delta':0.98}

stanrunmodel = load_or_compile_stan_model(model_name, force_recompile=False)

# # Load data from Gerkin scrape
DF = pd.read_csv(csv)

# t0 := where to start time series, index space
t0 = np.where(DF["new_cases"].values>=10)[0][0]
# tm := start of mitigation, index space
tm = t0 + 10

stan_data = {}
stan_data['n_scale'] = 1000 #use this instead of population
stan_data['n_theta'] = 8
stan_data['n_difeq'] = 5
stan_data['n_ostates'] = 3
stan_data['t0'] = t0-1 #to for ODE is one day, index before start of series
stan_data['tm'] = tm
stan_data['ts'] = np.arange(t0,len(DF['dates2']))
stan_data['y'] = (DF[['new_cases','new_recover','new_deaths']].to_numpy()).astype(int)[t0:,:]
stan_data['n_obs'] = len(DF['dates2']) - t0
# stan_data['rel_tol'] = 1e-2
# stan_data['max_num_steps'] = 1000

## function used to initialize parameters
def init_fun():
        x = {'theta':
             [np.random.lognormal(np.log(0.4),1)]
             + [np.random.lognormal(np.log(0.1),1)]
             + [np.random.lognormal(np.log(0.1),.5)]
             + [np.random.lognormal(np.log(0.1),.5)]
             + [np.random.lognormal(np.log(0.1),.5)]
             + [np.random.lognormal(np.log(0.001),1)]
             + [np.random.lognormal(np.log(0.5),.2)]
             + [np.random.lognormal(np.log(stan_data['tm']),.2)]
             #[np.random.lognormal(np.log(.01),1)]
            }
        return x

# ## Fit Stan
fit = stanrunmodel.sampling(data=stan_data, init=init_fun, control=control, chains=n_chains, chain_id=np.arange(n_chains),
                            warmup=n_warmups, iter=n_iter, thin=n_thin)

## uncomment to print fit summary
print(fit)

## save fit
save_dir = Path(__file__).parent / "fits"
# save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / ("model_fit_%s_%s.pkl" % (model_name, roi))
with open(save_path, "wb") as f:
    pickle.dump({'model_name' : model_name, 'model_code': stanrunmodel.model_code, 'fit' : fit}, f, protocol=-1)
