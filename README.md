# NIDDK SICR Model for estimating SARS-CoV-2 infection in a population

Code and data used for Chow et al, "TITLE", to model the progression of the COVID-19 epidemic and estimate the unobserved SARS-CoV-2 infected population from reported cases, case recoveries, and case deaths globablly.  Models are implemented in Stan and fit using PyStan.  Several versions of latent variable SIR models are provided in the `models` directory.

*PYPI and other badges here*

*BASIC SUMMARY OF THE MODEL HERE*

Data sources include the Johns Hopkins 

### Requirements to run code here:
- python 3.5+
- Install the Python package:
  - From pypi: `pip install niddk-sicr-covid19`
  - Or from source:
    - `git clone http://github.com/ccc1685/covid-19`
    - `cd covid-19`
    - `pip install -e .`
  - Or place this directory in your `PYTHONPATH` and install the contents of `requirements.txt`. 
- `pystan` may not be easily pip installable on some systems, so consider conda:
  - `conda install -c conda-forge pystan`

### Important scripts:
- Stan models can be run with Python file `scripts/run.py`:
  - Run a single region with:
    - `python scripts/run.py MODEL_NAME --roi=REGION_NAME`
    - e.g. `python scripts/run.py reducedlinearmodelR0 --roi=US_MI`
  - Other optional arguments for specifying paths and some fitting parameters can be examined with `python scripts/run.py --help`.
  - A pickle file containing the resultant fit will be produced in your `fits_path` (see help).

- Analyze finished fits for all regions with `scripts/visualize-master.py`:
  - For all regions (with fits) with `python scripts/visualize_master.py --model_name=MODEL_NAME`
  - e.g. `python visualize_master.py --model_name=reducedlinearmodelR0`
  - As above, help is available with the `--help` flag.
  - Jupyter notebooks containining all analyzed regions will be created in your `fits_path`.

- Tables summarizing fit parameters can be generated with `scripts/make-tables.py`:
  - `python scripts/make-tables.py`
  - e.g. `python scripts/make-tables.py --model-names reducedlinearmodelR0 fulllinearmodel`
  - As above, help is available with the `--help` flag.
  - `.csv` files of the resulting dataframes will be created in the `tables` directory (see help).

This code is open source under the MIT License.
Correspondence on modeling should be directed to *CARSON and SHASHAANK*
Correspondence on the python code should be directed to rgerkin at asu.edu.
