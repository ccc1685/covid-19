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
args = parser.parse_args()


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
df.to_csv('n_data.csv')