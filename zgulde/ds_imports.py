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
import math
from math import sqrt, factorial

import pandas as pd
from pandas import DataFrame, Series
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

from scipy import stats
from scipy.stats import binom, norm, geom, poisson

import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans, dbscan, k_means
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.pipeline import Pipeline

from pydataset import data

from zgulde import partition, chunk, comp, pluck, extend_pandas
from zgulde.ds_util import *

# check if we're running interactively
if hasattr(sys, 'ps1'):
    # turn on interactive mode in matplotlib
    plt.ion()

r = MyRange()
tips = data('tips')
mpg = data('mpg')
mtcars = data('mtcars')
swiss = data('swiss')
iris = data('iris').cleanup_column_names()

# plotting style defaults
plt.rc('patch', edgecolor='black', force_edgecolor=True, facecolor='firebrick')
plt.rc('axes', grid=True)
plt.rc('grid', linestyle=':', linewidth=.8, alpha=.7)
plt.rc('axes.spines', right=False, top=False)
plt.rc('figure', figsize=(11, 8))
plt.rc('font', size=12.0)
plt.rc('hist', bins=25)
