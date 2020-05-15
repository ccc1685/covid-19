#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from multiprocessing import Pool
import pandas as pd
import papermill as pm
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
for lib in ['blib2to3', 'papermill']:
    logger = logging.getLogger(lib)
    logger.setLevel(logging.WARNING)

from niddk_covid_sicr import get_data_prefix, get_ending, list_rois

notebook_path = Path(__file__).parent.parent / 'notebooks'

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description=('Executes all of the analysis '
                                              'notebooks'))

parser.add_argument('model_name',
                    help='Name of the Stan model file (without extension)')
parser.add_argument('-dp', '--data_path', default='./data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits_path', default='./fits',
                    help='Path to directory containing pickled fit files')
parser.add_argument('-rp', '--results_path', default='./results/vis-notebooks',
                    help=('Path to directory where resulting notebooks '
                          'will be stored'))
parser.add_argument('-mp', '--models_path', default='./models',
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
    if '_path' in key and 'results' not in key:
        assert Path(value).is_dir(),\
            "%s is not a directory" % Path(value).resolve()

# pathlibify some paths
data_path = Path(args.data_path)
fits_path = Path(args.fits_path)
models_path = Path(args.models_path)
results_path = Path(args.results_path)
results_path.mkdir(parents=True, exist_ok=True)

assert any([x.name.endswith('.csv') for x in data_path.iterdir()]),\
    "No .csv files found in data_path %s" % (data_path.resolve())
assert any([x.name.endswith('.stan') for x in models_path.iterdir()]),\
    "No .stan files found in models_path %s" % (models_path.resolve())
assert any([x.name.endswith('.pkl') or x.name.endswith('.csv')
            for x in fits_path.iterdir()]),\
    "No .pkl or .csv files found in fits_path %s" % (fits_path.resolve())

ending = get_ending(args.fit_format)

if not args.rois:
    data_rois = list_rois(data_path, get_data_prefix(), '.csv')
    fit_rois = list_rois(fits_path, args.model_name, ending)
    args.rois = list(set(data_rois).intersection(fit_rois))

args.n_threads = min(args.n_threads, len(args.rois))

print("Running visualization notebook for %d rois on model '%s'" %
      (len(args.rois), args.model_name))

# Make sure all ROI pickle files exist
for roi in args.rois:
    file = fits_path / ('%s_%s%s' % (args.model_name, roi, ending))
    assert file.is_file(), "No such %s file: %s" % (ending, file.resolve())


# Function to be execute on each ROI
def execute(model_name, roi, data_path, fits_path, model_path, notebook_path,
            results_path, fit_format, verbose=False):
    try:
        result = pm.execute_notebook(
            str(notebook_path / 'visualize.ipynb'),
            str(results_path / ('visualize_%s_%s.ipynb' % (model_name, roi))),
            parameters={'model_name': model_name,
                        'roi': roi,
                        'data_path': str(data_path),
                        'fits_path': str(fits_path),
                        'models_path': str(models_path),
                        'fit_format': fit_format},
            nest_asyncio=True)
    except pm.PapermillExecutionError as e:
        exception = '%s: %s' % (e.ename, e.evalue)
    except Exception as e:
        exception = str(e.split('\n')[-1:])
    else:
        # Possible exception that was raised
        # (or `None` if notebook completed successfully)
        exception = str(result['metadata']['papermill']['exception'])
    if exception and verbose:
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
                               models_path, notebook_path, results_path,
                               args.fit_format],
                              {'verbose': args.verbose},
                              callback=update)
        for roi in args.rois}
pool.close()

pool.join()
print('\n')
error_table = pd.Series({roi: job.get() for roi, job in jobs.items()})
error_table = error_table[error_table != 'None']
if len(error_table):
    print("Errors:")
    print(error_table)
