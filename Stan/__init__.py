import os
import pandas as pd
from pathlib import Path
import pickle
import platform
import pystan
import re
import sys

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


def list_rois(path, prefix, extension):
    """List all of the ROIs for which there is data or fits
    with ending `ending` at the given path.
    Assumes there are no underscores until immediately before
    the region begins in each potential file name"""
    if isinstance(path, str):
        path = Path(path)
    rois = []
    for file in path.iterdir():
        file_name = str(file.name)
        if file_name.startswith(prefix) and file_name.endswith(extension):
            roi = file_name.replace(prefix, '').replace(extension, '').strip('.').strip('_')
            rois.append(roi)
    return rois


def load_fit(fit_path, model_full_path, new_module_name=None):
    """This function will try to load a pickle file containing a Stan fit instance (and other things)
    If the compiled model is not found in memory, preventing the fit instance from being loaded, 
    it will trick Stan into thinking that the the model that is loaded is the model that belongs with that fit instance
    Then it will return samples"""
    try:
        with open(fit_path, "rb") as f:
            fit = pickle.load(f)
    except ModuleNotFoundError as e:
        matches = re.findall("No module named '([a-z0-9_]+)'", str(e))
        if matches:
            model = load_or_compile_stan_model(model_full_path)  # Load the saved, compiled model (or compile it)
            old_module_name = [x for x in sys.modules if 'stanfit4' in x][0]  # Get the name of the loaded model module in case we need it
            new_module_name = matches[0]
            sys.modules[new_module_name] = sys.modules[old_module_name]
            fit = load_fit(fit_path, old_module_name, new_module_name)
        else:
            raise ModuleNotFoundError("Module not found message did not parse correctly")
    else:
        fit = fit['fit']
    return fit


def extract_samples(fits_path, models_path, model_name, roi, fit_format):
    """Extract samples from the fit into a dataframe"""
    if fit_format in [0]:
        # Load the format that is just samples in a .csv file
        fit_path = Path(fits_path) / ("%s_%s.csv" % (model_name, roi))
        samples = pd.read_csv(fit_path)
    elif fit_format==1:
        # Load the format that is a pickle fit containing a Stan fit instance and some other things
        fit_path = Path(fits_path) / ("%s_%s.pkl" % (model_name, roi))
        model_full_path = Path(models_path) / model_name
        fit = load_fit(fit_path, model_full_path)
        samples = fit.to_dataframe()
    return samples


def make_table(roi, samples, params, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], chain=None):
    new_params = []
    for param in params:
        found = False
        if param in samples:
            new_params.append(param)
            found = True
        else:  # If not found in pure form
            for i in range(1000, 0, -1):  # Search for it in series from latest to earliest
                candidate = '%s[%d]' % (param, i)
                if candidate in samples:
                    new_params.append(candidate)  # Pick the latest one found
                    found = True
                    break  # And move on
        if not found:
            print("No param like %s is in the samples dataframe" % param)
    
    dfs = []
    cols = [col for col in samples if col in new_params]
    if chain:
        samples = samples[samples['chain']==chain]
    samples = samples[cols]
    q = samples.quantile(quantiles)#.reset_index(drop=True)
    dfs.append(q)
    df = pd.concat(dfs)
    df.columns = [x.split('[')[0] for x in df.columns]  # Drop the index
    df.index = pd.MultiIndex.from_product(([roi], df.index), names=['roi', 'quantile'])
    return df