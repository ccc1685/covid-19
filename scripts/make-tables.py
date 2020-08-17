#!/usr/bin/env python
# coding: utf-8

import argparse
from datetime import datetime
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

parser.add_argument('-ms', '--model-names',
                    default=[], nargs='+',
                    help=('Name of the Stan model files '
                          '(without .stan extension)'))
parser.add_argument('-mp', '--models-path', default='./models',
                    help='Path to directory containing the .stan model files')
parser.add_argument('-dp', '--data-path', default='./data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits-path', default='./fits',
                    help='Path to directory to save fit files')
parser.add_argument('-tp', '--tables-path', default='./tables/',
                    help='Path to directory to save tables')
parser.add_argument('-f', '--fit-format', type=int, default=1,
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
parser.add_argument('-ft', '--fixed-t', type=int, default=0,
                    help=('Use a fixed time base (where 1/22/20 is t=0)'
                          'rather than a time base that is relative to the '
                          'beginning of the data for each region'))
parser.add_argument('-ao', '--average-only', type=int, default=0,
                    help=('Assume all of the model-specific tables already '
                          'exist, skip creating them and instead make only '
                          'the concatenated (raw) and reweighted tables'))
args = parser.parse_args()

# If no model_names are provided, use all of them
if not args.model_names:
    args.model_names = ncs.list_models(args.models_path)
    assert len(args.model_names),\
        ("No such model files matching: *.stan' at %s" % (args.models_path))

# Get all model_names, roi combinations
if not args.average_only:
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
    if args.fixed_t:
        args.roi = roi  # Temporary
        csv = Path(args.data_path) / ("covidtimeseries_%s.csv" % args.roi)
        csv = csv.resolve()
        assert csv.exists(), "No such csv file: %s" % csv
        stan_data, t0 = ncs.get_stan_data(csv, args)
        global_start = datetime.strptime('01/22/20', '%m/%d/%y')
        frame_start = datetime.strptime(t0, '%m/%d/%y')
        day_offset = (frame_start - global_start).days
    else:
        day_offset = 0
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
                        stats, quantiles=args.quantiles,
                        day_offset=day_offset)
    return model_name, roi, df


tables_path = Path(args.tables_path)
tables_path.mkdir(exist_ok=True)

if not args.average_only:
    result = p_map(roi_df, repeat(args), *combos)

dfs = []
for model_name in args.model_names:
    out = tables_path / ('%s_fit_table.csv' % model_name)
    if not args.average_only:
        tables = [df_ for model_name_, roi, df_ in result
              if model_name_ == model_name]
        if not len(tables):  # Probably no matching models
            continue
        df = pd.concat(tables)
        df = df.sort_index()
        # Export the CSV file for this model
        df.to_csv(out)
    else:
        try:
            df = pd.read_csv(out)
        except FileNotFoundError:
            print('No table found for %s; skipping...' % model_name)
            continue
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
n_data_path = Path(args.data_path) / ('n_data.csv')
if n_data_path.resolve().is_file():
    extra = pd.read_csv(n_data_path).set_index('roi')
    extra['t0'] = extra['t0'].fillna('2020-01-23').astype('datetime64').apply(lambda x: x.dayofyear).astype(int)
    # Model-averaged table
    ncs.reweighted_stats(out, extra=extra, dates=args.dates)
else:
    print("No sample size file found at %s; unable to compute global average" % n_data_path.resolve())
