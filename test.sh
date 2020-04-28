pip install -e .
python scripts/get-data.py
# This is a totally insufficient number of samples to draw 
# It is only this slow here to support continuous integration,
# and should not be used for actual fits (which should use a
# much larger number of warmup samples (e.g. 5000) and
# iterations (e.g. 10000)
python scripts/run.py nonlinearmodel --n-warm=5 --n-iter=50
python scripts/visualize.py nonlinearmodel
python scripts/make-tables.py

