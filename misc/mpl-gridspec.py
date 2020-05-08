from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydataset
plt.ion()
mpg = pydataset.data('mpg')


fig = plt.figure(figsize=(8, 8))
gs = plt.GridSpec(3, 3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[1:, 2])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
plt.show()

fig = plt.figure(figsize=(13, 7))
gs = plt.GridSpec(6, 1)
ax1 = fig.add_subplot(gs[:5, :])
ax2 = fig.add_subplot(gs[5, :])
ax1.hist(mpg.hwy)

