import pystan, pickle, sys
import pandas as pd
import numpy as np
from __init__ import load_or_compile_stan_model

roi = sys.argv[1]
#expects data to be in local folder called data
fname = "../data/covid_timeseries_"+roi+".csv"

n_chains=4
n_warmups=5000
n_iter=10000
n_thin=1
control = {'adapt_delta':0.98}


stan_name = 'reducedlinearmodel'
stanrunmodel = load_or_compile_stan_model(stan_name, force_recompile=False)

# # Load data from Gerkin scrape

DF = pd.read_csv(fname)

# t0 := where to start time series, index space
t0 = np.where(DF["new_cases"].values>=10)[0][0]
# tm := start of mitigation, index space
tm = t0 + 10


# ## Format JHU ROI data for Stan

# int<lower = 1> n_obs;       // number of days observed
# int<lower = 1> n_theta;     // number of model parameters
# int<lower = 1> n_difeq;     // number of differential equations for yhat
# int<lower = 1> n_ostates;     // number of observed states
# int<lower = 1> n_pop;       // population
# real<lower = 1> n_scale;       // scale to match observed scale
# int y[n_obs,n_ostates];           // data, per-day-tally [cases,recovered,death]
# real t0;                // initial time point
# real tm; //start day of mitigation
# real ts[n_obs];         // time points that were observed
# int<lower = 1> n_obs_predict;       // number of days to predict
# real ts_predict[n_obs_predict];         //

stan_data = {}

# stan_data['n_pop'] = pop[roi]
stan_data['n_scale'] = 100000#10000000 #use this instead of population

stan_data['n_theta'] = 9
stan_data['n_difeq'] = 5
stan_data['n_ostates'] = 3

stan_data['t0'] = t0-1 #to for ODE is one day, index before start of series
stan_data['tm'] = tm
stan_data['ts'] = np.arange(t0,len(DF['dates2']))
# DF = DF.replace('NaN', 0)
stan_data['y'] = (DF[['new_cases','new_recover','new_deaths']].to_numpy()).astype(int)[t0:,:]
# stan_data['y'][stan_data['y']<0] = 0
stan_data['n_obs'] = len(DF['dates2']) - t0


# theta[1] ~ lognormal(log(0.25),1); //beta
# theta[2] ~ lognormal(log(0.1),1); //sigmac
# theta[3] ~ lognormal(log(0.01),1); //sigmar
# theta[4] ~ lognormal(log(0.01),1); //sigmad
# theta[5] ~ lognormal(log(0.01),1); //q
# theta[6] ~ lognormal(log(1),1); //sigmau
# theta[7] ~ lognormal(log(0.1),1); //mbase
# theta[8] ~ lognormal(log(tm),5); //mlocation
# theta[9] ~ lognormal(log(1),1);// theta_init


## function used to initialize parameters
def init_fun():
        x = {'theta':
             [np.random.lognormal(np.log(0.3),2)]+
             [np.random.lognormal(np.log(0.2),2)]+
             [np.random.lognormal(np.log(0.1),2)]+
             [np.random.lognormal(np.log(0.1),2)]+
             [np.random.lognormal(np.log(0.01),2)]+
             [np.random.lognormal(np.log(0.1),1)]+
             [np.random.lognormal(np.log(0.1),1)]+
             [np.random.lognormal(np.log(stan_data['tm']),3)]+
             [np.random.lognormal(np.log(1),.2)]
            }
        return x


# ## Fit Stan

fit = stanrunmodel.sampling(data = stan_data,init = init_fun ,control=control, chains = n_chains,chain_id=np.arange(n_chains), warmup = n_warmups, iter = n_iter, thin=n_thin, seed=13219)

## save fit
# import pickle
# with open("model_fit_"+roi+".pkl", "wb") as f:
    # pickle.dump({'model' : stanrunmodel, 'fit' : fit}, f, protocol=-1)

## uncomment to print fit summary
print(fit)
