#!/usr/bin/env python
# coding: utf-8

import argparse
from itertools import repeat
import pandas as pd
from pathlib import Path
from p_tqdm import p_map
import warnings
warnings.simplefilter("ignore")

import niddk_covid_sicr as ncs

# Parse all the command-line arguments
parser = argparse.ArgumentParser(description=('Generates an all-regions table '
                                              'for a model'))

parser.add_argument('-ms', '--model_names',
                    default=[], nargs='+',
                    help=('Name of the Stan model files '
                          '(without .stan extension)'))
parser.add_argument('-mp', '--models_path', default='./models',
                    help='Path to directory containing the .stan model files')
parser.add_argument('-fp', '--fits_path', default='./fits',
                    help='Path to directory to save fit files')
parser.add_argument('-tp', '--tables_path', default='./tables/',
                    help='Path to directory to save tables')
parser.add_argument('-f', '--fit_format', type=int, default=1,
                    help='Version of fit format')
parser.add_argument('-p', '--params', default=['R0', 'car', 'ifr'], nargs='+',
                    help='Which params to include in the table')
parser.add_argument('-d', '--dates', default=None, nargs='+',
                    help='Which dates to include in the table')
parser.add_argument('-ql', '--quantiles',
                    default=[0.025, 0.25, 0.5, 0.75, 0.975], nargs='+',
                    help='Which quantiles to include in the table ([0-1])')
parser.add_argument('-r', '--rois', default=[], nargs='+',
                    help=('Which rois to include in the table '
                          '(default is all of them)'))
parser.add_argument('-a', '--append', type=int, default=0,
                    help='Append to old tables instead of overwriting them')
args = parser.parse_args()

# If no model_names are provided, use all of them
if not args.model_names:
    args.model_names = ncs.list_models(args.models_path)
    assert len(args.model_names),\
        ("No such model files matching: *.stan' at %s" % (args.models_path))

# Get all model_names, roi combinations
combos = []
for model_name in args.model_names:
    model_path = ncs.get_model_path(args.models_path, model_name)
    extension = ['csv', 'pkl'][args.fit_format]
    rois = ncs.list_rois(args.fits_path, model_name, extension)
    if args.rois:
        rois = list(set(rois).intersection(args.rois))
    combos += [(model_name, roi) for roi in rois]
# Organize into (model_name, roi) tuples
combos = list(zip(*combos))
assert len(combos), "No combinations of models and ROIs found"
print("There are %d combinations of models and ROIs" % len(combos))


def roi_df(args, model_name, roi):
    model_path = ncs.get_model_path(args.models_path, model_name)
    extension = ['csv', 'pkl'][args.fit_format]
    rois = ncs.list_rois(args.fits_path, model_name, extension)
    if args.rois:
        rois = list(set(rois).intersection(args.rois))
    fit_path = ncs.get_fit_path(args.fits_path, model_name, roi)
    if args.fit_format == 1:
        fit = ncs.load_fit(fit_path, model_path)
        stats = ncs.get_waic_and_loo(fit)
        samples = fit.to_dataframe()
    elif args.fit_format == 0:
        samples = ncs.extract_samples(args.fits_path, args.models_path,
                                      model_name, roi, args.fit_format)
        stats = ncs.get_waic(samples)
    df = ncs.make_table(roi, samples, args.params,
                        stats, quantiles=args.quantiles)
    return model_name, roi, df


result = p_map(roi_df, repeat(args), *combos)
tables_path = Path(args.tables_path)
tables_path.mkdir(exist_ok=True)

dfs = []
for model_name in args.model_names:
    tables = [df_ for model_name_, roi, df_ in result
              if model_name_ == model_name]
    if not len(tables):  # Probably no matching models
        continue
    df = pd.concat(tables)
    out = tables_path / ('%s_fit_table.csv' % model_name)
    df = df.sort_index()
    # Export the CSV file for this model
    df.to_csv(out)
    # Then prepare for the big table (across models)
    df['model'] = model_name
    dfs.append(df)

# Raw table
df = pd.concat(dfs).reset_index().\
        set_index(['model', 'roi', 'quantile']).sort_index()
out = tables_path / ('fit_table_raw.csv')

# Possibly append
if args.append and out.is_file():
    try:
        df_old = pd.read_csv(out, index_col=['model', 'roi', 'quantile'])
    except:
        print("Cound not read old fit_table_raw file; overwriting it.")
    else:
        df = pd.concat([df_old, df])

df = df[sorted(df.columns)]

# Remove duplicate model/region combinations (keep most recent)
df = df[~df.index.duplicated(keep='last')]

# Export the CSV file for the big table
df.to_csv(out)

# Get n_data_pts and t0 obtained from `scripts/get-n-data.py`
path = Path(args.fits_path) / ('n_data.csv')
print(path)
if path.is_file():
    extra = pd.read_csv('n_data.csv').set_index('roi')
    extra['t0'] = extra['t0'].astype('datetime64').apply(lambda x: x.dayofyear).astype(int)

    # Model-averaged table
    ncs.reweighted_stats(out, extra=extra, dates=args.dates)
