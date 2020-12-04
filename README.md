# NIDDK SICR Model for estimating SARS-CoV-2 infection in a population

Code and data used for Chow et al, ["Global prediction of unreported SARS-CoV2 infection from observed COVID-19 cases"](https://www.medrxiv.org/content/10.1101/2020.04.29.20083485v1), to model the progression of the COVID-19 epidemic and estimate the unobserved SARS-CoV-2 infected population from reported cases, case recoveries, and case deaths globablly.  Models are implemented in Stan and fit using PyStan.  

[![PyPI version](https://badge.fury.io/py/covid-sicr.svg)](https://badge.fury.io/py/covid-sicr)
[![Build Status](https://travis-ci.org/nih-niddk-mbs/covid-sicr.svg?branch=refactor)](https://travis-ci.org/nih-niddk-mbs/covid-sicr)

The core model is a variation of the SIR model with a latent variable `I` for the number of *unobserved* infected which is distinguished from `C` the *observed* cases.  This model follows:

![formula](https://render.githubusercontent.com/render/math?math=\frac{dS}{dt}=-\frac{\beta}{N}S(I%2BqC))

![formula](https://render.githubusercontent.com/render/math?math=\frac{dI}{dt}=\frac{\beta}{N}S(I%2BqC)-\sigma_CI-\sigma_UI)

![formula](https://render.githubusercontent.com/render/math?math=\frac{dC}{dt}=\sigma_CI-\sigma_RC-\sigma_DC)

Several variants of this model are discussed in the Supplemental Material of the preprint.  The code required to fit these models to data is provided in the `models` directory. Data sources include The Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) and The COVID Tracking Project.

### Requirements to run code here:
- python 3.5+
- Install the Python package:
  - From pypi: `pip install covid-sicr`
  - Or from source:
    - `git clone http://github.com/nih-niddk-mbs/covid-sicr`
    - `cd covid-sicr`
    - `pip install -e .  # This installs the covid-sicr package from source`
  - Or place this directory in your `PYTHONPATH` and install the contents of `requirements.txt`.
- `pystan` may not be easily pip installable on some systems, so consider conda:
  - `conda install -c conda-forge pystan`

### Important scripts:
- New data can be downloaded from data sources with `scripts/get-data.py`:
  - For all data sources: `python scripts/get-data.py`.
  - This will use Johns Hopkins and COVID Tracking by default.  
  - Other options can be seen with the `--help` flag.
  - Data sources follow a functional pattern and are extensible.

- Stan models can be run with Python file `scripts/run.py`:
  - Run a single region with:
    - `python scripts/run.py MODEL_NAME --roi=REGION_NAME`
    - e.g. `python scripts/run.py SICRLMQ --roi=US_MI`
  - Other optional arguments for specifying paths and some fitting parameters can be examined with `python scripts/run.py --help`.
  - A pickle file containing the resultant fit will be produced in your `fits-path` (see help).
  - A `scripts/run-many.py` file is provided for reference but much better performance will be obtained by running `scripts/run.py` on a cluster.

- Analyze finished fits for all regions with `scripts/visualize.py`:
  - For all regions (with fits) with `python scripts/visualize.py MODEL_NAME`
  - e.g. `python visualize-master.py --nonlinearmodel`
  - As above, help is available with the `--help` flag.
  - Jupyter notebooks containining all analyzed regions will be created in your `--fits-path`.

- Tables summarizing fit parameters can be generated with `scripts/make-tables.py`:
  - `python scripts/make-tables.py`
  - e.g. `python scripts/make-tables.py --model-names nonlinearmodel fulllinearmodel`
  - As above, help is available with the `--help` flag.
  - `.csv` files of the resulting dataframes will be created in the `--fit-path` directory in the `tables` subdirectory.

Reproducbility of a particular version of the manuscript will require using a particular tag, e.g. ["medrXiv-v1"](https://github.com/nih-niddk-mbs/covid-sicr/tree/medRxiv-v1) (see tags and releases here).  Using the most recent version of any other branch (including master) may lead to different results as our research is on-going.

This code is open source under the MIT License.
Correspondence on modeling should be directed to carsonc at nih dot gov or vattikutis at mail dot nih dot gov.
Correspondence on the python code should be directed to rgerkin at asu dot edu.

AS testing branch..
