import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ion()

np.random.seed(123)
df = pd.DataFrame(
    dict(x=np.random.choice(list("ABCDE"), 1000), y=np.random.normal(0, 10, 1000))
)
s = df.groupby("x").y.mean()


def label_bars(ax: plt.Axes, number_format=".3f", **kwargs) -> plt.Axes:
    for patch in ax.patches:
        height = patch.get_height()
        text_x = patch.get_x() + (patch.get_width() / 2)
        text_y = height + (0.03 * height * -1)
        print("height:", height)
        print("text_y:", text_y)
        text = format(height, number_format)
        ax.text(
            text_x,
            text_y,
            text,
            **{
                "ha": "center",
                "va": "top" if height > 0 else "bottom",
                "c": "white",
                **kwargs,
            },
        )
    return ax


ax = s.plot.bar(width=0.9)
label_bars(ax, number_format=".2f")
