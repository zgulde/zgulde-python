"""
A module that provides all the common data science imports.

For example:

>>> from zgulde.ds_imports import *

and all the common data science imports will be in your namespace.

Intended for quick iteration / experimentation; explicit imports should be used
in production code.
"""

import itertools as it
import re
import sys
from functools import partial, reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as r
import seaborn as sns
import sklearn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame, Series
from pydataset import data
from scipy import stats
from scipy.stats import binom, geom, norm, poisson
from sklearn.cluster import DBSCAN, KMeans, dbscan, k_means
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import zgulde.extend_pandas
from zgulde.ds_util import *
from zgulde.util import *

# check if we're running interactively
if hasattr(sys, "ps1"):
    # turn on interactive mode in matplotlib
    plt.ion()

tips = data("tips")
mpg = data("mpg")
mtcars = data("mtcars")
swiss = data("swiss")
iris = data("iris").cleanup_column_names()

# plotting style defaults
plt.rc("patch", edgecolor="black", force_edgecolor=True, facecolor="firebrick")
plt.rc("axes", grid=True, fc="#FEFEFE")
plt.rc("grid", linestyle=":", linewidth=0.8, alpha=0.7)
plt.rc("axes.spines", right=False, top=False)
plt.rc("figure", figsize=(11, 8), fc="#FEFEFE")
plt.rc("font", size=12.0)
plt.rc("hist", bins=25)
