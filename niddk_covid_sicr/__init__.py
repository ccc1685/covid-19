"""The NIDDK SICR model for estimating the fraction infected with SARS-CoV-2"""

import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

from .io import *
from .stats import *
from .analysis import *
from .data import *
from .prep import *
from .prepV import *
from .prepX import *
