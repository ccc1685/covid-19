#!/usr/bin/env python
# coding: utf-8

import argparse
import blib2to3
import logging
from multiprocessing import Pool, TimeoutError
import os
import papermill as pm
from pathlib import Path
import subprocess
import sys
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
for lib in ['blib2to3', 'papermill']:
    logger = logging.getLogger(lib)
    logger.setLevel(logging.WARNING)

#papermill_logger = logging.getLogger('papermill')
#papermill_logger.setLevel(logging.WARNING)

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description='Executes all of the analysis notebooks')

parser.add_argument('-m', '--model_name', default='reducedlinearmodelR0',
                    help='Name of the Stan model file (without extension)')
parser.add_argument('-dp', '--data_path', default='../data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits_path', default='./fits',
                    help='Path to directory containing pickled fit files')
parser.add_argument('-pp', '--package_path', default='.',
                    help='Path to our python package (that contains __init__.py)')
parser.add_argument('-mp', '--models_path', default='.',
                    help='Path to directory containing .stan files')
parser.add_argument('-r', '--rois', default=[], nargs='+',
                    help='Space separated list of ROIs')
parser.add_argument('-n', '--n_threads', type=int, default=16, nargs='+',
                    help='Number of threads to use for analysis')
parser.add_argument('-f', '--fit_format', type=int, default=1,
                    help='Version of fit format')
parser.add_argument('-v', '--verbose', type=int, default=0,
                    help='Verbose error reporting')
args = parser.parse_args()

for key, value in args.__dict__.items():
    if '_path' in key:
        assert Path(value).is_dir(), "%s is not a directory" % Path(value).resolve()

# pathlibify some paths
data_path = Path(args.data_path)
fits_path = Path(args.fits_path)
models_path = Path(args.models_path)
package_path = Path(args.package_path)

assert any([x.name.endswith('.csv') for x in data_path.iterdir()]),\
    "No .csv files found in data_path %s" % (data_path.resolve())
assert any([x.name.endswith('.stan') for x in models_path.iterdir()]),\
    "No .stan files found in models_path %s" % (models_path.resolve())
assert any([x.name == '__init__.py' for x in package_path.iterdir()]),\
    "No __init__.py file found in package_path %s" % (package_path.resolve())
assert any([x.name.endswith('.pkl') or x.name.endswith('.csv') for x in fits_path.iterdir()]),\
    "No .pkl or .csv files found in package_path %s" % (fits_path.resolve())
    
sys.path.insert(0, str(package_path.resolve()))
from __init__ import list_rois, get_data_prefix, get_ending

ending = get_ending(args.fit_format)

if not args.rois:
    data_rois = list_rois(data_path, get_data_prefix(), '.csv')
    fit_rois = list_rois(fits_path, args.model_name, ending)
    print(data_rois, fit_rois)
    args.rois = list(set(data_rois).intersection(fit_rois))
        
print("Running visualization notebook for %d rois" % len(args.rois))

# Make sure all ROI pickle files exist
for roi in args.rois:
    file = fits_path / ('%s_%s%s' % (args.model_name, roi, ending))
    assert file.is_file(), "No such %s file: %s" % (ending, file.resolve())

# Say what we are doing
print("Analyzing fits for model %s using the %d ROIs selected at %s" % (args.model_name, len(args.rois), args.fits_path))

# Function to be execute on each ROI
def execute(model_name, roi, data_path, fits_path, package_path, model_path, fit_format, verbose=False):
    try:
        result = pm.execute_notebook(
        'visualize.ipynb',
            str(fits_path / ('visualize_%s_%s.ipynb' % (model_name, roi))),
            parameters={'model_name': model_name,
                        'roi': roi,
                        'data_path': str(data_path),
                        'fits_path': str(fits_path),
                        'package_path': str(package_path),
                        'models_path': str(models_path), 
                        'fit_format': fit_format},
            nest_asyncio=True)
    except Exception as e:
        exception = e
    else:
        # Possible exception that was raised (or `None` if notebook completed successfully)
        exception = result['metadata']['papermill']['exception']
    if exception == 'None':
        exception = None
    if verbose and exception:
        print(roi, exception)
    return exception

# Top progress bar (how many ROIs have finished)
pbar = tqdm(total=len(args.rois), desc="All notebooks", leave=True)
def update(*a):
    pbar.update()

# Execute up to 16 ROIs notebooks at once
pool = Pool(processes=args.n_threads)
jobs = {roi: pool.apply_async(execute,
                              [args.model_name, roi, data_path, fits_path,
                               package_path, models_path, args.fit_format],
                              {'verbose': args.verbose},
                              callback=update)
        for roi in args.rois}
pool.close()

# Check to see how many have finished.  
def check_status(rois, verbose=False):
    finished = []
    unfinished = []
    failed = []
    for roi in rois:
        try:
            exception = jobs[roi].get(timeout=1)
        except TimeoutError:
            unfinished.append(roi)
        else:
            if exception is None:
                finished.append(roi)
            else:
                failed.append('%s: %s' % (roi, exception))
    #clear_output()
    if verbose:
        print("Finished: %s" % ','.join(finished))
        print("=====")
        print("Unfinished: %s" % ','.join(unfinished))
        print("=====")
        print("Failed: %s" % ','.join(failed))
    return len(unfinished)

#check_status(args.rois)
# Wait for all jobs to finish
#while True:
#    if not check_status(args.rois, verbose=True):
#        break
pool.join()
print('')
