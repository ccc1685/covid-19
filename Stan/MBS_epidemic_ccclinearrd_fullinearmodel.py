import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pystan

def go(stanrunmodel, roi):
    DF = pd.read_csv("../data/covid_timeseries_"+roi+".csv")
    t0 = np.where(DF["new_cases"].values>=1)[0][0] 
    tm = t0 + 5 #np.where(mitigate[roi]==DF['dates2'])[0][0]

    stan_data = {}
    stan_data['n_scale'] = 100000#10000000 #use this instead of population
    stan_data['n_theta'] = 12
    stan_data['n_difeq'] = 2
    stan_data['n_ostates'] = 3
    stan_data['t0'] = t0-1 #to for ODE is one day, index before start of series
    stan_data['tm'] = tm
    stan_data['ts'] = np.arange(t0,len(DF['dates2'])) 
    stan_data['y'] = (DF[['new_cases','new_recover','new_deaths']].to_numpy()).astype(int)[t0:,:]
    stan_data['n_obs'] = len(DF['dates2']) - t0
    init = [{'theta':[0.25,0.1,0.01,0.01,0.01,1.0,0.1,1.0,1.0,0.1,10.0,1.0]}]

    def init_fun():
            x = {'theta':
                 [np.random.lognormal(np.log(0.25),1)]+
                 [np.random.lognormal(np.log(0.1),1)]+
                 [np.random.lognormal(np.log(0.01),1)]+
                 [np.random.lognormal(np.log(0.01),1)]+
                 [np.random.lognormal(np.log(0.01),1)]+
                 [np.random.lognormal(np.log(1),1)]+
                 [np.random.lognormal(np.log(0.1),1)]+
                 [np.random.lognormal(np.log(stan_data['tm']),5)]+
                 [np.random.lognormal(np.log(1),5)]+
                 [np.random.lognormal(np.log(0.1),1)]+
                 [np.random.lognormal(np.log(10),1)]+
                 [np.random.lognormal(np.log(1),1)]
                }
            return x

    n_chains=1
    n_warmups=1000
    n_iter=10000
    n_thin=50

    control = {'adapt_delta':0.99}
    fit = stanrunmodel.sampling(data = stan_data,init = init_fun ,control=control, chains = n_chains, warmup = n_warmups, iter = n_iter, thin=n_thin, seed=13219)

    with open("./fits/model_fit_%s.pkl" % roi, "wb") as f:
        pickle.dump({'model' : stanrunmodel, 'fit' : fit}, f, protocol=-1)
