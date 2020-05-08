import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def to_radians(a):
    'scales a from 0 to 2*pi'
    return minmax_scale(a.reshape(-1, 1), (0, 2 * np.pi)).ravel()


df = pd.DataFrame({
    'hour': np.arange(24),
    'coffee': sorted(np.random.uniform(0, 5, 24)),
})
df['hour_radians'] = to_radians(df.hour.values)
df

fig = plt.figure()
gs = plt.GridSpec(3, 3)
ax1 = fig.add_subplot(gs[-1, :])
ax2 = fig.add_subplot(gs[:-1, :], polar=True)
#
df.set_index('hour').coffee.plot.bar(ax=ax1, width=.9, ec='black')
# df.set_index('hour_radians').coffee.plot.bar(ax=ax2)
ax2.plot(df.hour_radians, df.coffee)
plt.setp(
    ax2,
    theta_direction=-1,
    theta_zero_location='N',
    xticks=df.hour_radians,
    xticklabels=df.hour,
)
# ax2.set_theta_direction(-1)
# ax2.set_theta_zero_location('N')

df

#
# plt.setp(ax2, xticks=to_radians(np.arange(25)), xticklabels=reversed(range(24)))
# ax.set_theta_zero_location('N')
# ax2.set_theta_offset(7 * ((2 * np.pi) / 24))
