"""Fit many models for many regions.  If you have HPC access it is recommended to 
instead use `run.py` in parallel."""

import argparse
import multiprocessing
import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import subprocess
from tqdm import tqdm

import niddk_covid_sicr as ncs

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description='Fits multiple Stan models for multiple regions')

parser.add_argument('-mn', '--model-names', default=[], nargs='+',
                    help='Name of the Stan model file (without extension)')
parser.add_argument('-mp', '--models-path', default='./models',
                    help='Path to directory containing the .stan model files')
parser.add_argument('-dp', '--data-path', default='./data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits-path', default='./fits',
                    help='Path to directory to save fit files')
parser.add_argument('-r', '--rois', default=[], nargs='+',
                    help='ROI to use')
parser.add_argument('-ch', '--n-chains', type=int, default=4,
                    help='Number of chains to run')
parser.add_argument('-wm', '--n-warmups', type=int, default=500,
                    help='Number of warmups')
parser.add_argument('-it', '--n-iter', type=int, default=1000,
                    help='Number of iterations')
parser.add_argument('-tn', '--n-thin', type=int, default=1,
                    help='thinning number')
parser.add_argument('-th', '--n-threads', type=int, default=0,
                    help='Number of threads to use the whole run')
parser.add_argument('-ad', '--adapt-delta', type=float, default=0.995,
                    help='Adapt delta control parameter')
parser.add_argument('-f', '--fit-format', type=int, default=1,
                    help='Version of fit format')
parser.add_argument('-i', '--init',
                    help=('Fit file to use for initial conditions '
                          '(uses last sample)'))
args = parser.parse_args()
if args.n_threads == 0:
    args.n_threads = args.n_chains
    
prefix = ncs.get_data_prefix()
if not args.rois:
    args.rois = ncs.list_rois(args.data_path, prefix, '.csv')
    assert len(args.rois),\
        ("No such data files matching: %s*.csv' at %s"
         % (prefix, args.data_path))

if not args.model_names:
    args.model_names = ncs.list_models(args.models_path)
    assert len(args.model_names),\
        ("No such model files matching: *.stan' at %s" % (args.models_path))
    
model_paths = [Path(args.models_path) / ('%s.stan' % model_name)
                for model_name in args.model_names]
for model_path in model_paths:
    assert model_path.is_file(), "No such .stan file: %s" % model_path

run_script_path = Path(__file__).parent / 'run.py'
run_flags = [('--%s' % key.replace('_', '-'), value) for key, value in args.__dict__.items()
             if key not in ['model_names', 'rois']]
    
# This next section will be run in serial since this is only a reference implementation
# Parallel implementations are possible with multiprocessing or the library of your choice
# Best performance comes from parallelizing run.py on a cluster.
i = 0
for model_name in tqdm(args.model_names, desc='Model'):
    for roi in tdqm(args.rois, desc='Region'):
        cmd = ['python', str(run_script_path.resolve()), model_name,
                '--roi', roi]
        for key, value in run_flags:
            cmd += [key, str(value)]
        subprocess.run(cmd)
        i += 1
        if i > 2:
            break
            
print("Finished all models and rois")
