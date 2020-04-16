#!/usr/bin/env python
# coding: utf-8

import argparse
from multiprocessing import Pool, TimeoutError
import os
import papermill as pm
from pathlib import Path
import subprocess
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description='Executes all of the analysis notebooks')

parser.add_argument('-m', '--model_name', default='reducedlinearmodelR0',
                    help='Name of the Stan model file (without extension)')
parser.add_argument('-dp', '--data_path', default='../data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits_path', default='./fits',
                    help='Path to directory containing pickled fit files')
parser.add_argument('-r', '--rois', default=[], nargs='+',
                    help='Space separated list of ROIs')
parser.add_argument('-n', '--n_threads', type=int, default=16, nargs='+',
                    help='Number of threads to use for analysis')
parser.add_argument('-f', '--fit_format', type=int, default=0,
                    help='Version of fit format')
args = parser.parse_args()

# pathlibify the fits_path            
fits_path = Path(args.fits_path)

# Set default ROIs if not provided
if args.fit_format in [0]:
    ending = '.csv'
elif args.fit_format==1:
    ending = '.pkl'
else:
    raise Exception("No such fit format: %s" % args.fit_format)
    
if not args.rois:
    for file in fits_path.iterdir():
        if str(file).endswith(ending):
            roi = '_'.join(file.name.split('.')[0].split('_')[1:])
            args.rois.append(roi)
            
print(args.rois)

# Make sure all ROI pickle files exist
for roi in args.rois:
    file = fits_path / ('%s_%s%s' % (args.model_name, roi, ending))
    assert file.is_file(), "No such %s file: %s" % (ending, file.resolve())

# Say what we are doing
print("Analyzing fits for model %s using the %d ROIs selected at %s" % (args.model_name, len(args.rois), args.fits_path))

# Function to be execute on each ROI
def execute(model_name, roi, data_path, fits_path, fit_format):
    os.makedirs('fits', exist_ok=True)
    result = pm.execute_notebook(
        'visualize.ipynb',
        str(fits_path / ('visualize_%s_%s.ipynb' % (model_name, roi))),
        parameters={'model_name': model_name,
                    'roi': roi,
                    'data_path': data_path,
                    'fits_path': str(fits_path),
                    'fit_format': fit_format},
        nest_asyncio=True)
    # Possible exception that was raised (or `None` if notebook completed successfully)
    exception = result['metadata']['papermill']['exception']
    return exception

# Top progress bar (how many ROIs have finished)
pbar = tqdm(total=len(args.rois), desc="All notebooks", leave=True)
def update(*a):
    pbar.update()

# Execute up to 16 ROIs notebooks at once
pool = Pool(processes=args.n_threads)
jobs = {roi: pool.apply_async(execute,
                              [args.model_name, roi, args.data_path, fits_path, args.fit_format],
                              callback=update)
        for roi in args.rois}
pool.close()

# Check to see how many have finished.  
def check_status(rois):
    finished = []
    unfinished = []
    failed = []
    for roi in rois:
        try:
            exception = jobs[roi].get(timeout=0.1)
        except TimeoutError:
            unfinished.append(roi)
        else:
            if exception is None:
                finished.append(roi)
            else:
                failed.append('%s: %s' % (roi, exception))
    #clear_output()
    print("Finished: %s" % ','.join(finished))
    print("=====")
    print("Unfinished: %s" % ','.join(unfinished))
    print("=====")
    print("Failed: %s" % ','.join(failed))
    return len(unfinished)

# Wait for all jobs to finish
#while True:
#    if not check_status(args.rois):
#        break
pool.join()
print('')
