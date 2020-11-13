"""Fit one model for a single region.  Typically used in a batch file
to run multiple regions and models at a time."""

import argparse
import numpy as np
from pathlib import Path
import pickle
import sys

import niddk_covid_sicr as ncs

# Parse all the command-line arguments
parser = argparse.ArgumentParser(
    description='Fits one Stan model for one region')

parser.add_argument('model_name',
                    help='Name of the Stan model file (without extension)')
parser.add_argument('-mp', '--models-path', default='./models',
                    help='Path to directory containing the .stan model files')
parser.add_argument('-dp', '--data-path', default='./data',
                    help='Path to directory containing the data files')
parser.add_argument('-fp', '--fits-path', default='./fits',
                    help='Path to directory to save fit files')
parser.add_argument('-r', '--roi', default='US_NY',
                    help='ROI to use')
parser.add_argument('-ch', '--n-chains', type=int, default=1,
                    help='Number of chains to run')
parser.add_argument('-wm', '--n-warmups', type=int, default=50,
                    help='Number of warmups')
parser.add_argument('-it', '--n-iter', type=int, default=100,
                    help='Number of iterations')
parser.add_argument('-tn', '--n-thin', type=int, default=1,
                    help='thinning number')
parser.add_argument('-th', '--n-threads', type=int, default=0,
                    help='Number of threads to use the whole run')
parser.add_argument('-ad', '--adapt-delta', type=float, default=0.85,
                    help='Adapt delta control parameter')
parser.add_argument('-fc', '--force-recompile', type=int, default=0,
                    help='Force recompilation of model (no cache)')
parser.add_argument('-f', '--fit-format', type=int, default=1,
                    help='Version of fit format')
parser.add_argument('-i', '--init',
                    help=('Fit file to use for initial conditions '
                          '(uses last sample)'))
parser.add_argument('-ld', '--last-date',
                    help=('Last date to use in the data; dates past this '
                          'will be ignored'))
parser.add_argument('-nd', '--n-data-only',
                    help=('Only calculate number of data points used for each'
                          'region, write to a table, and stop before fitting'))
parser.add_argument('-ft', '--fixed-t', type=int, default=0,
                    help=('Use a fixed time base (where 1/22/20 is t=0)'
                          'rather than a time base that is relative to the '
                          'beginning of the data for each region'))
args = parser.parse_args()

if args.n_threads == 0:
    args.n_threads = args.n_chains
if args.n_iter < args.n_warmups:
    args.n_warmups = int(args.n_iter/2)

csv = Path(args.data_path) / ("covidtimeseries_%s.csv" % args.roi)
csv = csv.resolve()
assert csv.exists(), "No such csv file: %s" % csv

stan_data, t0 = ncs.get_stan_data(csv, args)
if stan_data is None:
    print("No data for %s; skipping fit." % args.roi)
    sys.exit(0)
if args.n_data_only:
    print(ncs.get_n_data(stan_data))
    sys.exit(0)
init_fun = ncs.get_init_fun(args, stan_data)

model_path = Path(args.models_path) / ('%s.stan' % args.model_name)
model_path = model_path.resolve()
assert model_path.is_file(), "No such .stan file: %s" % model_path

control = {'adapt_delta': args.adapt_delta}
stanrunmodel = ncs.load_or_compile_stan_model(args.model_name,
                                              args.models_path,
                                              force_recompile=args.force_recompile)

# Fit Stan
# fit = stanrunmodel.sampling(data=stan_data, init=init_fun, control=control,
#                             chains=args.n_chains,
#                             chain_id=np.arange(args.n_chains),
#                             warmup=args.n_warmups, iter=args.n_iter,
#                             thin=args.n_thin)
fit = stanrunmodel.vb(data=stan_data,init=init_fun)

# Uncomment to print fit summary
print(fit)

# Save fit
save_dir = Path(args.fits_path)
save_dir.mkdir(parents=True, exist_ok=True)
if args.fit_format == 0:
    save_path = save_dir / ("%s_%s.csv" % (args.model_name, args.roi))
    result = fit.to_dataframe().to_csv(save_path)
else:
    save_path = save_dir / ("%s_%s.pkl" % (args.model_name, args.roi))
    with open(save_path, "wb") as f:
        pickle.dump({'model_name': args.model_name,
                     'model_code': stanrunmodel.model_code, 'fit': fit},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

print("Finished %s" % args.roi)
