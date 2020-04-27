"""Loading, saving, and listing of data and fits."""

import os
import pandas as pd
from pathlib import Path
import pickle
import platform
import pystan
import re
import sys


def get_fit_path(fits_path: str, model_name: str, roi: str) -> str:
    """Get a full path contain a model fit for one region.

    Args:
        fits_path: Path to directory where fits are stored.
        model_name: Name of the model (without .stan suffix).
        roi: A single region of interest, e.g. "US_MI" or "Greece".

    Returns:
        A full path to a model fit for one region.
    """
    path = Path(fits_path)
    path = path / ('%s_%s.pkl' % (model_name, roi))
    assert path.is_file(), "No pickled fit file found at %s" % path
    return path


def get_model_path(models_path: str, model_name: str) -> str:
    """Get a full model path for one model file.

    Args:
        models_path: Path to directory where models are stored.
        model_name: Name of the model (without .stan suffix).

    Returns:
        A full path to a Stan model file.
    """
    file_path = Path(models_path) / ('%s.stan' % model_name)
    assert file_path.is_file(), "No .stan file found at %s" % file_path
    path = Path(models_path) / model_name
    return path


def get_data(roi: str, data_path: str = 'data') -> pd.DataFrame:
    """Get the data associated with a given ROI.

    Args:
        roi (str): A single region of interest, e.g. "US_MI" or "Greece".
        data_path (str, optional): A path to the directory where data
                                   is stored.

    Returns:
        pd.DataFrame: A dataframe containing the data.
    """
    path = Path(data_path) / ("covidtimeseries_%s.csv" % roi)
    assert path.is_file(), "No file found at %s" % (path.resolve())
    df = pd.read_csv(path).set_index('dates2')
    df = df[[x for x in df if 'Unnamed' not in x]]
    df.index.name = 'date'
    return df


def load_or_compile_stan_model(stan_name: str, force_recompile: bool = False,
                               verbose: bool = False):
    """[summary]

    Args:
        stan_name (str): [description]
        force_recompile (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    stan_raw = '%s.stan' % stan_name
    stan_compiled = '%s_%s_%s.stanc' % (stan_name, platform.platform(),
                                        platform.python_version())
    stan_raw_last_mod_t = os.path.getmtime(stan_raw)
    try:
        stan_compiled_last_mod_t = os.path.getmtime(stan_compiled)
    except FileNotFoundError:
        stan_compiled_last_mod_t = 0
    if force_recompile or (stan_compiled_last_mod_t < stan_raw_last_mod_t):
        sm = pystan.StanModel(file=stan_raw)
        with open(stan_compiled, 'wb') as f:
            pickle.dump(sm, f)
    else:
        if verbose:
            print("Loading %s from cache..." % stan_name)
        with open(stan_compiled, 'rb') as f:
            sm = pickle.load(f)
    return sm


def get_data_prefix() -> str:
    """[summary]

    Returns:
        str: [description]
    """
    return 'covidtimeseries'


def get_ending(fit_format: int) -> str:
    """[summary]

    Args:
        fit_format (int): [description]

    Raises:
        Exception: [description]

    Returns:
        str: [description]
    """
    if fit_format in [0]:
        ending = '.csv'
    elif fit_format == 1:
        ending = '.pkl'
    else:
        raise Exception("No such fit format: %s" % fit_format)
    return ending


def list_rois(path: str, prefix: str, extension: str) -> list:
    """List all of the ROIs for which there is data or fits.

    Restricts to those with ending `ending` at the given path. Assumes there
    are no underscores until immediately before the region begins in each
    potential file name

    Args:
        path (str): [description]
        prefix (str): [description]
        extension (str): [description]

    Returns:
        list: [description]
    """
    if isinstance(path, str):
        path = Path(path)
    rois = []
    for file in path.iterdir():
        file_name = str(file.name)
        if file_name.startswith(prefix+'_') and file_name.endswith(extension):
            roi = file_name.replace(prefix, '').replace(extension, '')\
                           .strip('.').strip('_')
            rois.append(roi)
    return rois


def load_fit(fit_path: str, model_full_path: str, new_module_name: str = None):
    """Return a Stan fit instance.

    This function will try to load a pickle file containing a Stan fit instance
    (and other things). If the compiled model is not found in memory,
    preventing the fit instance from being loaded, it will make Stan think that
    the the model that is loaded is the model that belongs with that fit
    instance. Then it will return samples.

    Args:
        fit_path (str): [description]
        model_full_path (str): [description]
        new_module_name (str, optional): [description]. Defaults to None.

    Raises:
        ModuleNotFoundError: [description]

    Returns:
        [type]: [description]
    """
    try:
        with open(fit_path, "rb") as f:
            fit = pickle.load(f)
    except ModuleNotFoundError as e:
        matches = re.findall("No module named '([a-z0-9_]+)'", str(e))
        if matches:
            # Load the saved, compiled model (or compile it)
            load_or_compile_stan_model(model_full_path)
            # Get the name of the loaded model module in case we need it
            old_module_name = [x for x in sys.modules if 'stanfit4' in x][0]
            new_module_name = matches[0]
            sys.modules[new_module_name] = sys.modules[old_module_name]
            fit = load_fit(fit_path, old_module_name, new_module_name)
        else:
            msg = "Module not found message did not parse correctly"
            raise ModuleNotFoundError(msg)
    else:
        fit = fit['fit']
    return fit


def extract_samples(fits_path: str, models_path: str, model_name: str,
                    roi: str, fit_format: int) -> pd.DataFrame:
    """Extract samples from the fit into a dataframe.

    Args:
        fits_path (str): [description]
        models_path (str): [description]
        model_name (str): [description]
        roi (str): [description]
        fit_format (int): [description]

    Returns:
        pd.DataFrame: [description]
    """
    if fit_format in [0]:
        # Load the format that is just samples in a .csv file
        fit_path = Path(fits_path) / ("%s_%s.csv" % (model_name, roi))
        samples = pd.read_csv(fit_path)
    elif fit_format == 1:
        # Load the format that is a pickle fit containing a Stan fit instance
        # and some other things
        fit_path = Path(fits_path) / ("%s_%s.pkl" % (model_name, roi))
        model_full_path = get_model_path(models_path, model_name)
        fit = load_fit(fit_path, model_full_path)
        samples = fit.to_dataframe()
    return samples


def last_sample_as_dict(fit_path: str, model_path: str) -> dict:
    """Return the last sample of a fit as a dict.

    For example for intializing a sampling session that starts from the last
    sample of a previous one.

    Args:
        fit_path (str): [description]
        model_path (str): [description]

    Returns:
        dict: [description]
    """
    fit = load_fit(fit_path, model_path)
    last = {key: value[-1] for key, value in fit.extract().items()}
    return last
