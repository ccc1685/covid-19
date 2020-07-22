import json
import pandas as pd


# Open JSON files containing countries and US ROIs
with open('countries.json') as country_file, open('USrois.json') as US_file:
    countries = json.load(country_file)
    US = json.load(US_file)

# Instantiate DataFrame
df = pd.DataFrame (columns = ['roi','region', 'subregion'])

# Fill DataFrame with countries
for entry in countries:

    roi = entry['name']['common']
    region = entry['region']
    subregion = entry['subregion']

    rois = {
             'roi': roi,
             'region': region,
             'subregion': subregion
             }
    df = df.append(rois, ignore_index=True)

for entry in US:
    roi = 'US_' + str(entry['abbreviation'])
    region = 'North America'
    subregion = 'United States'

    rois = {
            'roi': roi,
            'region': region,
            'subregion': subregion
            }
    df = df.append(rois, ignore_index=True)

# Export DataFrame to CSV
df.to_csv('rois.csv', index=True)
