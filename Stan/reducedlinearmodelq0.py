import numpy as np

def init_func(stan_data):  # new format
    return dict(f1=np.random.gamma(1.5,2.),
                f2=np.random.gamma(1.5,1.5),
                sigmar=np.random.gamma(2.,.1/2.),
                sigmad=np.random.gamma(2.,.1/2.),
                sigmau=np.random.gamma(2.,.1/2.),
                mbase=np.random.gamma(2.,.1/2.),
                mlocation=np.random.lognormal(np.log(stan_data['tm']),1.),
                extra_std=np.random.exponential(1.)
                )
