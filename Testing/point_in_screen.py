from itertools import islice, cycle

from tqdm import tqdm
import sys
import cv2
import numpy as np
from sklearn import cluster, datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics

from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path
from sklearn.preprocessing import StandardScaler

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")
plt.xticks(())
plt.yticks(())

#Screen Co-ords taken for testing
Screen = [[0.5132, 0.5131], [0.5468, 0.2821], [0.23140000000000005, 0.2167], [0.2136, 0.5]]

#Checking paths
p = path.Path(Screen)
points = np.array([[0.5100, 0.5100],
                  [0.5232, 0.5231],
                  [0.5232, 0.5731],
                  [0.5232, 0.5231],
                  [0.5032, 0.5231],
                  [0.8232, 0.5231],
                  [0.5232, 0.6231],
                  [0.5232, 0.5231],
                  [0.7232, 0.5231],
                  [0.5232, 0.5231],
                  [0.1232, 0.5231]])
#print(p.contains_points(points))

#Clustering




np.random.seed(0)

n_samples = 300

#Generating random datasets
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.08)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.08)

eps = 0.22

#Joining the datasets
datasets = [
    noisy_circles,
    noisy_moons,
]
#dataset
X = datasets[1][0]
X=StandardScaler().fit_transform(X)

#Model
dbscan = DBSCAN(eps=eps,min_samples=4)
#Fitting
model = dbscan.fit(X)

labels = model.labels_

sample_cores = np.zeros_like(labels,dtype=bool)

sample_cores[dbscan.core_sample_indices_]=True

n_clusters=len(set(labels))-(1 if -1 in labels else 0)

#print(n_clusters)
#print(X)
label_iter = 0
#print(sys.getsizeof(X))
#print(sys.getsizeof(X[0]))

for _,label in tqdm(enumerate(labels),desc="Drawing_Clusters",total=len(labels)):
    #print(label)
    #print(labels[0])
    #print(len(X))
    #print(len(labels))
    y_pred = dbscan.labels_.astype(np.int)
    colors = np.array(list(islice
                           (cycle
                                  (["#FE4A49",
                                    "#2AB7CA",
                                    "#A1C38C",
                                    "#666699",
                                    "#efe464",
                                    "#b24c37",
                                    "#432eb7",
                                    "#3c8608",
                                    "#d2c815",
                                    "#d1aa99",
                                    "#09d788",
                                    "#3c028f",
                                    "#27fa97",
                                    "#9764c5",
                                    "#09c8f1",
                                    "#034d24",
                                    "#84e332",
                                    "#a7cd76",
                                    "#6d73b6",
                                    "#2F847C"]), n_clusters+1)))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[y_pred])
    label_iter += 1
plt.show()
