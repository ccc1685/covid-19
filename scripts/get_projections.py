import pandas as pd
import io
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.stats.proportion as smp
import pylab
from scipy.optimize import curve_fit
import niddk_covid_sicr as ncs
import requests
import bz2
import pickle
import _pickle as cPickle

import bz2
import pickle
import _pickle as cPickle

# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

fitloc = '/data/carsonc/covid-19/fitUS06-27-20'
rois = ncs.list_rois(fitloc, 'SICRMQC2R2DX2', '.pkl')

for current_roi in rois:

    df = ncs.extract_samples(fitloc, './models/', 'SICRMQC2R2DX2', current_roi, 1)
    proj_cols = [col for col in df.columns if '_proj' in col]
    compressed_pickle(current_roi,df[proj_cols])
    print(current_roi,len(proj_cols))
