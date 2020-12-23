"""The NIDDK SICR model for estimating the fraction infected with SARS-CoV-2"""
import os
import numexpr
# numexpr.set_num_threads(numexpr.detect_number_of_cores())
ncpus = numexpr.detect_number_of_cores()
if 'SLURM_CPUS_PER_TASK' in os.environ:
     try:
         ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
     except ValueError:
         ncpus = 2
elif 'SLURM_CPUS_ON_NODE' in os.environ:
     try:
         ncpus = int(os.environ['SLURM_CPUS_ON_NODE'])
     except ValueError:
         ncpus = 2

from .io import *
from .stats import *
from .analysis import *
from .data import *
from .prep import *
from .prepV import *
