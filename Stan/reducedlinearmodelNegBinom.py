import numpy as np

# functions used to initialize parameters
def init_func(stan_data):  # old format
    x = {'theta':
    # Numpy convention: gamma(shape,scale)
        [np.random.gamma(1.5,2.)]    #f1
      + [np.random.gamma(1.5,1.5)]   #f2
      + [np.random.gamma(2.,.1/2)]   #sigmar
      + [np.random.gamma(2.,.1/2)]   #sigmad
      + [np.random.gamma(2.,.1/2)]   #sigmau
      + [0.*np.random.exponential(.01)]  #q
      + [np.random.exponential(.5)]  #mbase
      + [np.random.lognormal(np.log(stan_data['tm']),.5)]   #mlocation
      + [np.random.exponential(1.)]   # extra_std
        }
    return x
