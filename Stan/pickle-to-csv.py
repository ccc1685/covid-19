from pathlib import Path
from unpickling_tools import unpickle_fit
import pickle
import pandas as pd
import pystan
import re
import sys
from tqdm import tqdm

model_name = sys.argv[1]
fits_dir = sys.argv[2]

fits_dir = Path(fits_dir)
for file in tqdm(fits_dir.iterdir()):
    if model_name in file.name and '.pkl' in file.name:
        roi = '_'.join(str(file).split('.')[0].split('_')[1:])
        print(roi)
        try:
            with open(file, "rb") as f:
                fit = pickle.load(f)
        except ModuleNotFoundError as e:
            matches = re.findall("No module named '([a-z0-9_]+)'", str(e))
            if matches:
                module_name = matches[0]
                from unpickling_tools import unpickle_fit, create_fake_model
                fit = unpickle_fit(str(file), module_name=module_name)
    csv_path = fits_dir / ("%s_%s.csv" % (model_name, roi))
    fit['fit'].to_dataframe().to_csv(csv_path)