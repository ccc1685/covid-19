#!/usr/bin/env python
# coding: utf-8

import argparse
from itertools import product, repeat
from multiprocessing import Pool
import pandas as pd
from pathlib import Path
from p_tqdm import p_map
import sys
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

# Get Stan directory onto path
path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(path))

import Stan as cs

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description='Generates an all-regions table for a model')

parser.add_argument('-ms', '--model_names', default=['reducedlinearmodelq0', 'reducedlinearmodelq0ctime', 'reducedlinearmodelNegBinom', 'fulllinearmodel'], nargs='+',
                    help='Name of the Stan model files (without .stan extension)')
parser.add_argument('-mp', '--models_path', default='.',
                    help='Path to directory containing the .stan model files')
parser.add_argument('-fp', '--fits_path', default='./fits',
                    help='Path to directory to save fit files')
parser.add_argument('-f', '--fit_format', type=int, default=1,
                    help='Version of fit format')
parser.add_argument('-p', '--params', default=['R0', 'car', 'ifr'], nargs='+',
                    help='Which params to include in the table')
parser.add_argument('-ql', '--quantiles', default=[0.025, 0.25, 0.5, 0.75, 0.975], nargs='+',
                    help='Which quantiles to include in the table ([0-1])')
parser.add_argument('-r', '--rois', default=[], nargs='+',
                    help='Which rois to include in the table (default is all of them)')
args = parser.parse_args()

# Get all model_names, roi combinations
combos = []
for model_name in args.model_names:
    model_path = cs.get_model_path(args.models_path, model_name)
    extension = ['csv', 'pkl'][args.fit_format]
    rois = cs.list_rois(args.fits_path, model_name, extension)
    combos += [(model_name, roi) for roi in rois]
# Organize into (model_name, roi) tuples
combos = list(zip(*combos))

def roi_df(args, model_name, roi):
    model_path = cs.get_model_path(args.models_path, model_name)
    extension = ['csv', 'pkl'][args.fit_format]
    rois = cs.list_rois(args.fits_path, model_name, extension)
    if args.rois:
        rois = list(set(rois.intersection(args.rois)))
    fit_path = cs.get_fit_path(args.fits_path, model_name, roi)
    if args.fit_format==1:
        fit = cs.load_fit(fit_path, model_path)
        stats = cs.get_waic_and_loo(fit)
        samples = fit.to_dataframe()
    elif args.fit_format==0:
        samples = cs.extract_samples(fits_path, models_path, model_name, roi, fit_format)
        stats = cs.get_waic(samples)
    df = cs.make_table(roi, samples, args.params, stats, quantiles=args.quantiles)
    return model_name, roi, df

#with Pool(2) as p:
#    result = p.starmap(roi_df, zip(repeat(args), *combos))
result = p_map(roi_df, repeat(args), *combos)
tables_path = Path(args.fits_path) / 'tables'
tables_path.mkdir(exist_ok=True)

dfs = []
for model_name in args.model_names:
    df = pd.concat([df_ for model_name_, roi, df_ in result if model_name_==model_name])
    out = tables_path / ('%s_fit_table.csv' % model_name)
    # Export the CSV file for this model
    df.to_csv(out)
    # Then prepare for the big table (across models)
    df['model'] = model_name
    median_locs = df.index.get_level_values('quantile')==0.5
    df = df[median_locs].droplevel('quantile')
    dfs.append(df)

df = pd.concat(dfs).reset_index().set_index(['model', 'roi']).sort_index()
out = tables_path / ('fit_table.csv')
# Export the CSV file for the big table
df.to_csv(out)
