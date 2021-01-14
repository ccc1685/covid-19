""" Get population counts for rois and add to data_path/population_estimates.csv """
""" data sources: https://github.com/samayo/country-json, worldometers """

import json
import csv
import pandas as pd
from pathlib import Path
import glob, os

data_path = Path('../data')
#
# df_pop = pd.read_json(data_path / 'country-by-population.json')
df_pop_est = pd.read_csv(data_path / 'population_estimates.csv')
# df_ts = pd.read_csv(data_path / 'timeseries_countries.csv')
#
# df_pop.rename(columns={'country':'roi'}, inplace=True)
# df_ts.rename(columns={'Countries':'roi'}, inplace=True)
#
#
# # Add countries from all population counts that match timeseries countries
# # df = df_ts.merge(df_pop, on='roi', how='outer')
# # df.to_csv(data_path / 'test.csv')
#
# df_countries_pop = pd.read_csv(data_path / 'test.csv')
# df_countries_states = pd.concat([df_pop_est, df_countries_pop], sort=False)
# df_countries_states.sort_values(by=['roi'], inplace=True)
# df_countries_states.to_csv(data_path / 'population_estimates.csv', index=False)

# Get list of all rois in data path so we can calculate set difference
rois_csvs = [os.path.basename(x)[16:-4] for x in glob.glob('../data/*') if 'covidtimeseries' in x]
rois_pop = df_pop_est.roi.values
print(rois_csvs)
print("**")
dif1 = list(set(rois_csvs) - set(rois_pop))
dif2 = list(set(rois_pop) - set(rois_csvs))

print(dif1, dif2)
#
# for file in glob.glob(data_path / 'covidtimeseries_'):
#     print(file)
