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


def get_model_path(models_path: str, model_name: str,
                   compiled: bool = False, with_suffix: bool = False,
                   check_exists: bool = True) -> str:
    """Get a full model path for one model file.

    Args:
        models_path: Path to directory where models are stored.
        model_name: Name of the model (without .stan suffix).

    Returns:
        A full path to a Stan model file.
    """
    models_path = Path(models_path)
    if compiled:
        file_path = models_path / ('%s_%s_%s.stanc' %
                                   (model_name, platform.platform(),
                                    platform.python_version()))
    else:
        file_path = Path(models_path) / ('%s.stan' % model_name)
    if check_exists:
        assert file_path.is_file(), "No %s file found at %s" %\
            ('.stanc' if compiled else '.stan', file_path)
    if not with_suffix:
        file_path = file_path.with_suffix('')
    return file_path.resolve()


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


def load_or_compile_stan_model(model_name: str, models_path: str = './models',
                               force_recompile: bool = False,
                               verbose: bool = False):
    """Loads a compiled Stan model from disk or compiles it if does not exist.

    Args:
        model_name (str): [description]
        force_recompile (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    uncompiled_path = get_model_path(models_path, model_name, with_suffix=True)
    compiled_path = get_model_path(models_path, model_name,
                                   compiled=True, with_suffix=True,
                                   check_exists=False)
    stan_raw_last_mod_t = os.path.getmtime(uncompiled_path)
    try:
        stan_compiled_last_mod_t = os.path.getmtime(compiled_path)
    except FileNotFoundError:
        stan_compiled_last_mod_t = 0
    if force_recompile or (stan_compiled_last_mod_t < stan_raw_last_mod_t):
        models_path = str(Path(models_path).resolve())
        sm = pystan.StanModel(file=str(uncompiled_path),
                              include_paths=[models_path])
        with open(compiled_path, 'wb') as f:
            pickle.dump(sm, f)
    else:
        if verbose:
            print("Loading %s from cache..." % model_name)
        with open(compiled_path, 'rb') as f:
            sm = pickle.load(f)
    return sm


def get_data_prefix() -> str:
    """A universal prefix for all data files.

    Returns:
        str: The prefix (data files will start with this and an underscore).
    """
    return 'covidtimeseries'


def get_ending(fit_format: int) -> str:
    """Get the file extension for a given fit format.

    Args:
        fit_format (int): .csv (0) or .pkl (1).

    Raises:
        Exception: If an invalid fit format is provided.

    Returns:
        str: The extension for the provided fit format.
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
        path (str): A fit_path or data_path contaning one file for each region.
        prefix (str): Restrict files to those starting with this string.
        extension (str): Restrict files to those ending with this string.

    Returns:
        list: All regions, e.g. ['US_MI', 'Greece', ...].
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


def list_models(models_path: str) -> list:
    """Lists all available Stan models.

    Args:
        models_path (str): Full path to the models directory.

    Returns:
        list: A list of Stan models (without '.stan' suffixes).
    """

    models_path = Path(models_path)
    model_paths = [file for file in models_path.iterdir()
                   if file.name[0] == file.name.upper()[0] and file.suffix == '.stan']
    models = [mp.with_suffix('').name for mp in model_paths]
    return models


def load_fit(fit_path: str, model_full_path: str, new_module_name: str = None):
    """Return a Stan fit instance.

    This function will try to load a pickle file containing a Stan fit instance
    (and other things). If the compiled model is not found in memory,
    preventing the fit instance from being loaded, it will make Stan think that
    the the model that is loaded is the model that belongs with that fit
    instance. Then it will return samples.

    Args:
        fit_path (str): Full path to the fit file.
        model_full_path (str): Full path to one model file.
        new_module_name (str, optional): Used only internally for recursion.
            Defaults to None.

    Raises:
        ModuleNotFoundError: This exception will be caught if the compiled
                             model is not already in memory, and will be
                             handled by loading it into memory and repeating.

    Returns:
        pystan.StanFit4model: A Stan fit instance.
    """
    try:
        with open(fit_path, "rb") as f:
            fit = pickle.load(f)
    except ModuleNotFoundError as e:
        matches = re.findall("No module named '([a-z0-9_]+)'", str(e))
        if matches:
            # Load the saved, compiled model (or compile it)
            models_path = str(Path(model_full_path).parent)
            model_name = Path(model_full_path).name.strip('.stan')
            load_or_compile_stan_model(model_name,
                                       models_path=models_path)
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
        fits_path (str): Full path to the fits directory.
        models_path (str): Full path to the models directory.
        model_name (str): Name of the model (without '.stan' extension).
        roi (str): A single region, e.g. "US_MI" or "Greece".
        fit_format (int): .csv (0) or .pkl (1).

    Returns:
        pd.DataFrame: Samples extracted from the fit instance.
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
        fit_path (str): Full path to the fit file.
        model_path (str): Full path to the model.

    Returns:
        dict: Parameter:value pairs from the last sample of the given fit.
    """
    fit = load_fit(fit_path, model_path)
    last = {key: value[-1] for key, value in fit.extract().items()}
    return last
