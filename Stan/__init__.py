import os
import pickle
import platform
import pystan

def load_or_compile_stan_model(stan_name, force_recompile=False):
    stan_raw = '%s.stan' % stan_name
    stan_compiled = '%s_%s_%s.stanc' % (stan_name, platform.platform(), platform.python_version())
    stan_raw_last_mod_t = os.path.getmtime(stan_raw) 
    try:
        stan_compiled_last_mod_t = os.path.getmtime(stan_compiled) 
    except FileNotFoundError:
        stan_compiled_last_mod_t = 0
    if force_recompile or (stan_compiled_last_mod_t < stan_raw_last_mod_t):
        sm = pystan.StanModel(file=stan_raw)#, verbose=True)
        with open(stan_compiled, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Loading %s from cache..." % stan_name)
        with open(stan_compiled, 'rb') as f:
            sm = pickle.load(f)
    return sm