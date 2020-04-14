#!/usr/bin/env python
# coding: utf-8

#from IP
from multiprocessing import Pool, TimeoutError
import os
import papermill as pm
from pathlib import Path
import subprocess
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

model_name = 'reducedlinearmodelR0'
fits_path = Path('/home/rgerkin/dev/covid-fits')

rois = []
for file in fits_path.iterdir():
    if 'US_' in str(file) and str(file).endswith('.pkl'):
        roi = '_'.join(file.name.split('.')[0].split('_')[3:])
        rois.append(roi)
print("There are %d ROIs: %s" % (len(rois), rois))

def execute(model_name, roi):
    os.makedirs('fits', exist_ok=True)
    result = pm.execute_notebook(
        'visualize.ipynb',
        str(fits_path / ('visualize_%s_%s.ipynb' % (model_name, roi))),
        parameters={'model_name': model_name, 'roi': roi},
        nest_asyncio=True)
    exception = result['metadata']['papermill']['exception']
    return exception

pool = Pool(processes=16)
jobs = {roi: pool.apply_async(execute, [model_name, roi]) for roi in rois}

def check_status():
    finished = []
    unfinished = []
    failed = []
    for roi in rois:
        try:
            exception = jobs[roi].get(timeout=0.5)
        except TimeoutError:
            unfinished.append(roi)
        else:
            if exception is None:
                finished.append(roi)
            else:
                failed.append('%s: %s' % (roi, exception))
    #clear_output()
    print("Finished: %s" % ','.join(finished))
    print("=====")
    print("Unfinished: %s" % ','.join(unfinished))
    print("=====")
    print("Failed: %s" % ','.join(failed))
    if len(unfinished):
        return 1
    else:
        return 0

while True:
    if not check_status():
        break