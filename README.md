# covid-19

Code and data to model the Covid-19 progression and estimate the unobserved infected population from reported cases, case recoveries, and case deaths globablly.  Original code in Julia used a Metropolis-Hastings MCMC to fit various latent variable SIR models to the observable data.  Current updated models are implemented in PyStan and described below.  Models all end in .Stan.



Stan models can be run with Python file run.py.

Run a single region with:

`python run.py MODEL_NAME --roi=REGION_NAME` e.g. `python run.py reducedlinearmodelR0 --roi=US_MI`


Other Optional arguments:
```
  -dp DATA_PATH, --data_path DATA_PATH
                        Path to directory containing the data files
  -fp FITS_PATH, --fits_path FITS_PATH
                        Path to directory to save fit files
  -ch N_CHAINS, --n_chains N_CHAINS
                        Number of chains to run
  -wm N_WARMUPS, --n_warmups N_WARMUPS
                        Number of chains to run
  -it N_ITER, --n_iter N_ITER
                        Number of chains to run
  -tn N_THIN, --n_thin N_THIN
                        Number of chains to run
  -th N_THREADS, --n_threads N_THREADS
                        Number of threads to use the whole run
  -ad ADAPT_DELTA, --adapt_delta ADAPT_DELTA
                        Adapt delta control parameter
  -f FIT_FORMAT, --fit_format FIT_FORMAT
                        Version of fit format to save (0 for csv of samples, 1 for pickle of fit instance)
```       

Analyze finished fits for all regions with:

`python visualize_master.py --model_name=MODEL_NAME` e.g. `python visualize_master.py --model_name=reducedlinearmodelR0`

Other optional arguments:
```
  -dp DATA_PATH, --data_path DATA_PATH
                        Path to directory containing the data files
  -fp FITS_PATH, --fits_path FITS_PATH
                        Path to directory containing pickled fit files
  -pp PACKAGE_PATH, --package_path PACKAGE_PATH
                        Path to our python package (that contains __init__.py)
  -mp MODELS_PATH, --models_path MODELS_PATH
                        Path to directory containing .stan files
  -r ROIS [ROIS ...], --rois ROIS [ROIS ...]
                        Space separated list of ROIs
  -n N_THREADS [N_THREADS ...], --n_threads N_THREADS [N_THREADS ...]
                        Number of threads to use for analysis
  -f FIT_FORMAT, --fit_format FIT_FORMAT
                        Version of fit format to load (0 for csv of samples, 1 for pickle of fit instance)
  -v VERBOSE, --verbose VERBOSE
                        Verbose error reporting
```
