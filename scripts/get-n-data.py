import argparse
import pandas as pd
from pathlib import Path
import niddk_covid_sicr as ncs

# Parse all the command-line arguments
parser = argparse.ArgumentParser(
    description='')
parser.add_argument('-dp', '--data-path', default='./data',
                    help='Path to directory containing the data files')
parser.add_argument('-ld', '--last-date',
                    help=('Last date to use in the data; dates past this '
                          'will be ignored'))
parser.add_argument('-ft', '--fixed-t', type=int, default=0,
                    help=('Use a fixed time base (where 1/22/20 is t=0)'
                          'rather than a time base that is relative to the '
                          'beginning of the data for each region'))
args = parser.parse_args()


data_path = Path(args.data_path).resolve()
assert data_path.exists(), "No such data path: %s" % data_path

# Get all model_names, roi combinations
rois = ncs.list_rois(args.data_path, 'covidtimeseries', '.csv')
df = pd.DataFrame(index=rois, columns=['n_data_pts'], dtype=int)
for roi in rois:
    csv = Path(args.data_path) / ("covidtimeseries_%s.csv" % roi)
    csv = csv.resolve()
    assert csv.exists(), "No such csv file: %s" % csv
    stan_data, t0 = ncs.get_stan_data(csv, args)
    n_data = ncs.get_n_data(stan_data)
    df.loc[roi, 'n_data_pts'] = int(n_data)
    df.loc[roi, 't0'] = t0
df.index.name = 'roi'
df['n_data_pts'] = df['n_data_pts'].astype(int)
df['t0'] = df['t0'].astype('datetime64')
df.to_csv(data_path / 'n_data.csv')
print("Wrote sample size file to %s." % (data_path / 'n_data.csv'))