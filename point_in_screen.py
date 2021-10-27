import cv2
import numpy as np

from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path

#Screen Co-ords taken for testing
Screen = [[0.5132, 0.5131], [0.5468, 0.2821], [0.23140000000000005, 0.2167], [0.2136, 0.5]]

p = path.Path(Screen)
points = np.array([[0.5100, 0.5100],
                  [0.5232, 0.5231]])
print(p.contains_points(points))