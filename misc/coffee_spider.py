from collections import OrderedDict
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight']  = 'bold'

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def spider_plot(scores, ax, max_score=.5, markers=5):
    #### setup
    num_points = len(scores)
    angle_step = (2*np.pi) / num_points

    ## draw background grid
    marker_style = dict(color='gray', linestyle=':', marker='o',
                        markersize=5, markerfacecolor='white')

    labels = list(scores.keys())

    for i in range(0, num_points):
        pts = np.linspace(0,max_score, markers)
        theta = -(angle_step*i)

        xs = []
        ys = []
        for p in pts:
            x,y = pol2cart(p, theta)
            xs.append(x)
            ys.append(y)

        ax.plot(xs, ys, zorder=0, **marker_style)

        ## text rotation : if between 90 and 270, it needs to be flipped
        ##                 and we want to anchor end of word not beginning
        rot = (theta * (180/np.pi))
        va, ha = 'center', 'left'
        label = labels[i]
        if (rot < -90.) and (rot > -270):
            rot += 180
            ha = 'right'
            label += '  ' # append space between marker and text
        else:
            label = '  ' + label # prepend space between marker and text

        ax.annotate(label, (x,y), rotation_mode='anchor', rotation=rot, verticalalignment=va, horizontalalignment=ha)

    ## draw polygons
    patches = []

    values = list(scores.values())
    values.append(values[0]) # so that we can wrap around at the end

    # colors
    cmap = matplotlib.cm.get_cmap('plasma',14)

    for i in range(0, num_points):
        pts = [(0,0), pol2cart(values[i],-(angle_step*i)), pol2cart(values[i+1],-(angle_step*(i+1)))]
        poly = Polygon(pts, closed=False, zorder=10)
        patches.append(poly)

    p = PatchCollection(patches, cmap=cmap)

    p.set_array(np.linspace(0,1,num_points-1))
    ax.add_collection(p)

    ax.set_xlim(-max_score-.1,max_score+.1)
    ax.set_ylim(-max_score-.1,max_score+.1)

    return ax

scores = OrderedDict([
    ('FLORAL', .1),
    ('CITRUS\n  FRUIT', .4),
    ('STONE\n  FRUIT', .1),
    ('BERRY\n  FRUIT', .4),
    ('SPICY', .1),
    ('NUTTY', 0.),
    ('CHOCOLATE', .1),
    ('CARAMEL', .4),
    ('MOUTHFEEL', .3),
    ('BALANCE', .4),
    ('AFTERTASTE', .4),
    ('CLEAN', .4),
    ('SWEET', .4),
    ('ACIDITY', .4),
])

## draw N length lines from center (can have 5 points on them with marker and line style)
fig, ax = plt.subplots(figsize=(8,8))

ax = spider_plot(scores, ax)

## remove axis labels
plt.axis('off')
plt.show()
