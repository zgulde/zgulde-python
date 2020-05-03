import scipy.stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.preprocessing
from zgulde import partition

plt.ion()

scalers = [
    ("minmax", sklearn.preprocessing.MinMaxScaler()),
    ("standard", sklearn.preprocessing.StandardScaler()),
    ("quantile", sklearn.preprocessing.QuantileTransformer()),
    ("power", sklearn.preprocessing.PowerTransformer()),
]

np.random.seed(123)
# a = skewness
df = pd.DataFrame({"x": scipy.stats.skewnorm(a=6, loc=50, scale=20).rvs(1000)})
df.x.plot.hist()


for name, scaler in scalers:
    df[name] = scaler.fit_transform(df[["x"]])

fig, axs = plt.subplots(2, 2)
for ax, (name, scaler) in zip(axs.ravel(), scalers):
    ax.scatter(df.x, df[name])
    ax.set(title=name)

fig, axs = plt.subplots(2, 2)
for ax, (name, scaler) in zip(axs.ravel(), scalers):
    df[name].plot.hist(ax=ax)

fig, axs = plt.subplots(4, 2)
for (ax1, ax2), (name, scaler) in zip(partition(axs.ravel(), 2), scalers):
    df[name].plot.hist(ax=ax1)
    ax1.set(title=name)
    ax2.scatter(df.x, df[name])
    ax2.set(title=name)

fig, axs = plt.subplots(2, 4)
fig.tight_layout()
for (ax1, ax2), (name, scaler) in zip(zip(axs.ravel()[:4], axs.ravel()[4:]), scalers):
    df[name].plot.hist(ax=ax1)
    ax1.set(title=name)
    ax2.scatter(df.x, df[name])
    ax2.set(title=name, xlabel="original x value", ylabel=f"{name} scaled")
axs[1][0].text(40, 1, r"\frac{x - x_{min}}{x_{max} - x_{min}}", va="top")
axs[1][1].text(45, 4, r"$\frac{x - \mu}{\sigma}$", va="top", size=16)
