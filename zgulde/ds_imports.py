'''
A module that provides all the common data science imports.

For example:

>>> from zgulde.ds_imports import *

and all the common data science imports will be in your namespace.

Intended for quick iteration / experimentation, and explicit imports should be
used in production code.
'''

import sys
import os
import itertools as it
from functools import reduce, partial

import pandas as pd
from pandas import DataFrame, Series
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

from scipy import stats
import statsmodels.api as sm

from pydataset import data

from zgulde import partition, chunk, comp, pluck, extend_pandas

# check if we're running interactively
if hasattr(sys, 'ps1'):
    # turn on interactive mode in matplotlib
    plt.ion()
